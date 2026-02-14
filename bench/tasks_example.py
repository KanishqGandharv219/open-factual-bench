from .schema import Task

EXAMPLE_TASKS = [
    Task(
        id="science_001",
        domain="science",
        question="What is the chemical symbol for water?",
        reference_answer="H2O",
        context=None,
        metadata={"source": "synthetic_demo_v1"}
    )
]
