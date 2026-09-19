from hansgpt_research.checkpoint_retention import prune_run_checkpoints


def test_keep_recent_archives_and_final_without_touching_other_files(tmp_path):
    run = tmp_path / "new_run"
    run.mkdir()
    for n in [100, 110, 120, 130, 140, 150, 155]:
        (run / f"positions_{n:09d}.pt").write_text("checkpoint")
    (run / "notes.txt").write_text("preserve")
    (tmp_path / "positions_000000010.pt").write_text("other run")
    deleted = prune_run_checkpoints(run, keep_recent=3, retain_every=100, final_position=155)
    assert sorted(deleted) == [f"positions_{n:09d}.pt" for n in [110, 120, 130]]
    expected_names = {"notes.txt", *[f"positions_{n:09d}.pt" for n in [100, 140, 150, 155]]}
    assert {p.name for p in run.iterdir()} == expected_names
    assert (tmp_path / "positions_000000010.pt").read_text() == "other run"
