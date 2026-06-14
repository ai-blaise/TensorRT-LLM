from types import SimpleNamespace

from tensorrt_llm._torch.disaggregation.base.transfer import (
    SessionArgsBase,
    SessionStatus,
)
from tensorrt_llm._torch.disaggregation.native.transfer import (
    RxSession,
    TaskStatus,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2


class _FakeTransferSession:

    def __init__(self, *, completed=False, failed=False, transferring=False):
        self._completed = completed
        self._failed = failed
        self._transferring = transferring

    def is_completed(self):
        return self._completed

    def has_failed(self):
        return self._failed

    def has_transferring_tasks(self):
        return self._transferring


class _FakeReceiver:

    def __init__(self):
        self.released = []
        self.cleared = []

    def release_hisparse_request(self, rid, *, force=False):
        self.released.append((rid, force))

    def clear_session(self, rid):
        self.cleared.append(rid)


def test_disagg_collect_done_defers_failed_sessions_with_active_writes():
    transceiver = KvCacheTransceiverV2.__new__(KvCacheTransceiverV2)

    completed, failed = transceiver._collect_done(
        {
            1: _FakeTransferSession(completed=True),
            2: _FakeTransferSession(failed=True, transferring=True),
            3: _FakeTransferSession(failed=True, transferring=False),
        },
        {},
    )

    assert completed == [1]
    assert failed == [3]


def test_rx_session_close_defers_hisparse_release_while_transferring():
    receiver = _FakeReceiver()
    params = SimpleNamespace(disagg_request_id=123, ctx_request_id=None)
    session = RxSession.__new__(RxSession)
    session._closed = False
    session._receiver = receiver
    session._aux_buffer = None
    session.aux_slot = None
    session.request_id = 123
    session._base_args = SessionArgsBase(params)
    session._kv_tasks = [SimpleNamespace(status=TaskStatus.TRANSFERRING)]
    session._terminal_status = None
    session._exception = None
    session._need_aux = False
    session._aux_status = TaskStatus.INIT

    assert session.close() is False
    assert session._closed is False
    assert receiver.released == []
    assert receiver.cleared == []

    session._kv_tasks[0].status = TaskStatus.ERROR
    session._terminal_status = SessionStatus.ERROR
    assert session.close() is True
    assert session._closed is True
    assert receiver.released == [(123, True)]
    assert receiver.cleared == [123]
