from api.core.ab_testing import ABTestingManager


def test_ab_assignment_deterministic():
    manager = ABTestingManager(experiment_name="exp", split=50)
    a1 = manager.assign_bucket("user-123")
    a2 = manager.assign_bucket("user-123")
    assert a1 == a2

