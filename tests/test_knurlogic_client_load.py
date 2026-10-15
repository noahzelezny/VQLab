"""knurlogic_client.load retries only refusals for room still being freed;
unload waits until the job is gone."""
from vqlab.core.box import knurlogic_client as kc


class _K:
    def __init__(self, answers, gone_after=0):
        self.answers, self.calls, self.polls, self.gone_after = list(answers), 0, 0, gone_after

    def load(self, **kw):
        self.calls += 1
        return self.answers.pop(0)

    def unload(self, **kw):
        self.unloaded = kw

    def state(self):
        self.polls += 1
        return {"models": [] if self.polls > self.gone_after else [{"job": "j1"}]}


def test_room_refusal_is_retried_until_it_places():
    K = _K([{"refused": "fit: x fits on no set of the answering Macs"}, {"job": "j1"}])
    sleeps = []
    assert kc.load(K, "x", machines=["fit"], sleep=sleeps.append) == {"job": "j1"}
    assert K.calls == 2 and sleeps == [30]


def test_other_refusals_are_returned_at_once():
    K = _K([{"refused": "no such model"}])
    assert kc.load(K, "x", machines=["fit"], sleep=lambda s: None)["refused"] == "no such model"
    assert K.calls == 1


def test_unload_waits_until_the_job_is_gone():
    K = _K([], gone_after=2)
    sleeps = []
    kc.unload(K, {"job": "j1"}, sleep=sleeps.append)
    assert K.unloaded == {"job": "j1"} and len(sleeps) == 2


def test_unload_a_single_mac_instance():
    K = _K([])
    kc.unload(K, {"job": None, "instance": "i7"}, sleep=lambda s: None)
    assert K.unloaded == {"instance": "i7"}
