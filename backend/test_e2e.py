"""End-to-end test: create task -> execute via adapter -> check results."""
import asyncio
import json
from pathlib import Path

from app.core.config import settings
from app.adapters.executor_adapter import CLIExecutorAdapter, ExecutionRequest


async def main():
    csv_file = "/home/dff652/dff_project/inference-platform/data/PI_20412.PV.csv"
    output_dir = "/home/dff652/dff_project/inference-platform/data/test_output"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Python path: {settings.OLD_PYTHON_PATH}")
    print(f"Project path: {settings.OLD_PROJECT_PATH}")
    print(f"Input: {csv_file}")
    print(f"Output: {output_dir}")
    print()

    request = ExecutionRequest(
        task_id=999,
        method="adtk_hbos",
        input_files=[csv_file],
        output_dir=output_dir,
        n_downsample=5000,
    )

    adapter = CLIExecutorAdapter()

    # Show the command that will be run
    cmd = adapter._build_command(request)
    print(f"Command: {' '.join(cmd)}")
    print()

    print("Executing...")
    result = await adapter.execute(request)

    print(f"\nSuccess: {result.success}")
    print(f"Return code: {result.return_code}")
    print(f"Result files: {result.result_files}")
    print(f"Annotation count: {len(result.annotations)}")

    if result.stderr:
        print(f"\nSTDERR (last 500 chars):\n{result.stderr[-500:]}")
    if result.stdout:
        print(f"\nSTDOUT (last 500 chars):\n{result.stdout[-500:]}")

    if result.result_files:
        for f in result.result_files[:2]:
            print(f"\n--- {f} ---")
            with open(f) as fh:
                data = json.load(fh)
                print(json.dumps(data, indent=2, ensure_ascii=False)[:500])


if __name__ == "__main__":
    asyncio.run(main())
