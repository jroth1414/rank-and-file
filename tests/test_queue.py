import sys

from scripts.queue import run_queue


def test_queue_skips_done_and_retries_once(tmp_path):
    (tmp_path / "runs" / "done_run").mkdir(parents=True)
    (tmp_path / "runs" / "done_run" / "DONE").write_text("x")
    py = sys.executable
    lines = [
        f'{py} -c "print(1)" --name done_run',
        f'{py} -c "import sys; sys.exit(1)" --name failing',
        f"{py} -c \"open(r'{tmp_path}/ok.txt','w').write('hi')\" --name ok",
    ]
    q = tmp_path / "q.txt"
    q.write_text("# comment\n" + "\n".join(lines) + "\n")
    summary = run_queue(q, runs_root=tmp_path / "runs")
    assert summary == {"done_run": "skipped", "failing": "failed", "ok": "ok"}
    assert (tmp_path / "ok.txt").read_text() == "hi"
    assert "failing" in (tmp_path / "runs" / "queue.log").read_text()


def test_queue_skips_results_json(tmp_path):
    (tmp_path / "runs" / "results_run").mkdir(parents=True)
    (tmp_path / "runs" / "results_run" / "results.json").write_text("{}")
    py = sys.executable
    lines = [
        f'{py} -c "print(1)" --name results_run',
    ]
    q = tmp_path / "q.txt"
    q.write_text("\n".join(lines) + "\n")
    summary = run_queue(q, runs_root=tmp_path / "runs")
    assert summary == {"results_run": "skipped"}
