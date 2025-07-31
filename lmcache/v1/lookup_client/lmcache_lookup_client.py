# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from typing import TYPE_CHECKING, Optional, OrderedDict
import threading

# Third Party
from vllm.utils import make_zmq_socket
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.rpc_utils import get_zmq_rpc_path_lmcache

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class LMCacheLookupClient(LookupClientInterface):
    """ZMQ-based lookup client that communicates with a lookup server."""

    def __init__(self, vllm_config: "VllmConfig", config: LMCacheEngineConfig):
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        rpc_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache_rpc_port", 0
        )
        self.tensor_parallel_size = vllm_config.parallel_config.tensor_parallel_size
        self.create_lookup_server_only_on_worker_0 = (
            config.extra_config
            and config.extra_config.get("create_lookup_server_only_on_worker_0", True)
        )
        if self.create_lookup_server_only_on_worker_0:
            socket_path = get_zmq_rpc_path_lmcache(vllm_config, rpc_port)
            self.socket = make_zmq_socket(
                self.ctx,
                socket_path,
                zmq.REQ,  # type: ignore[attr-defined]
                bind=False,
            )
            return
        for tp_rank in range(self.tensor_parallel_size):
            socket_path = get_zmq_rpc_path_lmcache(vllm_config, rpc_port, tp_rank)
            if tp_rank == 0:
                self.socket = make_zmq_socket(
                    self.ctx,
                    socket_path,
                    zmq.REQ,  # type: ignore[attr-defined]
                    bind=False,
                )
            else:
                self.socket.connect(socket_path)

    def lookup(self, token_ids: torch.Tensor, request_id: Optional[str] = None, tags: OrderedDict = None) -> int:
        token_bufs = self.encoder.encode(token_ids)
        request_id_buf = request_id.encode("utf-8")
        tags_str = ""
        if tags is not None:
            tags_str = "@".join([f"{k}%{v}" for k, v in tags.items()])
        tags_buf = tags_str.encode("utf-8")
        if self.create_lookup_server_only_on_worker_0:
            self.socket.send_multipart(token_bufs + [request_id_buf, tags_buf], copy=False)
            resp = self.socket.recv()
            result = int.from_bytes(resp, "big")
            return result
        results = []
        for i in range(self.tensor_parallel_size):
            self.socket.send_multipart(token_bufs + [request_id_buf, tags_buf], copy=False)
            resp = self.socket.recv()
            result = int.from_bytes(resp, "big")
            results.append(result)
        if not all(x == results[0] for x in results):
            raise RuntimeError(
                f"Lookup results (number of hit tokens) differ "
                f"across tensor parallel ranks: {results}."
            )
        return results[0]

    def supports_producer_reuse(self) -> bool:
        """Return True as LMCacheLookupClient supports producer kvcache reuse"""
        return True

    def close(self):
        self.socket.close(linger=0)


class LMCacheLookupServer:
    """ZMQ-based lookup server that handles lookup requests using LMCacheEngine."""

    def __init__(self, lmcache_engine: LMCacheEngine, vllm_config: "VllmConfig"):
        self.decoder = MsgpackDecoder(torch.Tensor)
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        rpc_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache_rpc_port", 0
        )
        socket_path = get_zmq_rpc_path_lmcache(
            vllm_config, rpc_port, vllm_config.parallel_config.rank
        )
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,
        )

        self.lmcache_engine = lmcache_engine
        self.running = True

        def process_request():
            while self.running:
                # try:
                # request = self.socket.recv()
                frames = self.socket.recv_multipart(copy=False)
                token_frames = frames[:-2]
                request_id = frames[-2].bytes.decode("utf-8")
                tags_str = frames[-1].bytes.decode("utf-8")
                tags = None
                if tags_str != "":
                    tags = OrderedDict()
                    tags_list = tags_str.split("@")
                    for kv in tags_list:
                        kvs = kv.split("%", 1)
                        if len(kvs) != 2:
                            raise ValueError("Unexpected tags_str: {tags_str}")
                        tags[kvs[0]] = kvs[1]

                token_ids = self.decoder.decode(token_frames)
                result = self.lmcache_engine.lookup(
                    token_ids, request_id=request_id, pin=True, tags=tags,
                )
                response = result.to_bytes(4, "big")
                self.socket.send(response)
                # except Exception as e:
                #    logger.error("Error in LMCache lookup server: %s", e)
                #    break
                # continue

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):
        self.socket.close(linger=0)
        # TODO: close the thread!
