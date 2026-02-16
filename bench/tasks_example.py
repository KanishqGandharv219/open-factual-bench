from .schema import Task

EXAMPLE_TASKS = [
    # Science - factual QA
    Task(
        id="science_001",
        domain="science",
        question="What is the chemical symbol for water?",
        reference_answer="H2O",
        context=None,
        source="synthetic_demo_v1",
        created_at="2025-02",
        notes="Basic science factual QA",
        metadata={"type": "factual_qa"}
    ),
    Task(
        id="science_002",
        domain="science",
        question="What year was the James Webb Space Telescope launched?",
        reference_answer="2021",
        context=None,
        source="post_2024_event",
        created_at="2025-02",
        notes="Simple space-history question; good for factual recall",
        metadata={"type": "factual_qa"}
    ),

    # Math - factual QA
    Task(
        id="math_001",
        domain="math",
        question="What is the derivative of x^2?",
        reference_answer="2x",
        context=None,
        source="synthetic_demo_v1",
        created_at="2025-02",
        notes="Symbolic differentiation, very common textbook example",
        metadata={"type": "factual_qa"}
    ),
    Task(
        id="math_002",
        domain="math",
        question="If a triangle has sides of length 3, 4, and 5, is it a right triangle?",
        reference_answer="Yes",
        context=None,
        source="synthetic_demo_v1",
        created_at="2025-02",
        notes="Classic 3-4-5 Pythagorean triple",
        metadata={"type": "factual_qa"}
    ),

    # Code - factual QA
    Task(
        id="code_001",
        domain="code",
        question="In Python, what keyword is used to define a function?",
        reference_answer="def",
        context=None,
        source="synthetic_demo_v1",
        created_at="2025-02",
        notes="Basic Python syntax recall",
        metadata={"type": "factual_qa"}
    ),

    # Current events - contamination-resistant
    Task(
        id="events_001",
        domain="current_events",
        question="Who won the 2026 Australian Open men's singles title?",
        reference_answer="[To be filled based on actual 2026 event]",
        context=None,
        source="post_training_cutoff_2026",
        created_at="2026-02",
        notes="Deliberately post-training-cutoff factual question",
        metadata={"type": "factual_qa"}
    ),

    # Hallucination stress-test (closed-book, expects refusal/uncertainty)
    Task(
        id="hallucination_001",
        domain="other",
        question="What is the capital city of the fictional country Westeros?",
        reference_answer="This is a fictional setting; there is no real capital. Models should refuse or acknowledge uncertainty.",
        context=None,
        source="synthetic_demo_v1",
        created_at="2025-02",
        notes="Designed to trigger hallucination unless model handles fiction vs reality",
        metadata={"type": "hallucination_stress"}
    ),
]
