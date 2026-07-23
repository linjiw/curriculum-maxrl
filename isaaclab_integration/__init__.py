"""Isaac Lab integration for the frontier_rl curriculum-MaxRL schedule.

Layout:
  frontier_terms.py       curriculum terms (teacher/scripted/uniform/static) as
                          ManagerTermBase subclasses — the registration idiom this
                          fork requires (see INTEGRATION.md §1).
  train_frontier.py       in-container launcher: stock rsl_rl train.py flow + --arm.
  test_frontier_terms.py  CPU tests against a stub env (no isaaclab import needed).

Import as ``isaaclab_integration.frontier_terms`` with
``scripts/curriculum-maxrl`` on sys.path (the launcher does this).
"""
