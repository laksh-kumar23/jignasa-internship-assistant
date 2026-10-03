"""Unit tests for BM25 retrieval module."""

from datetime import date
import pytest

from schemas import Notice
from retrieval.bm25 import BM25Retriever, tokenize


@pytest.fixture
def sample_notices() -> list[Notice]:
    """Fixture providing a diverse set of sample internship notices."""
    return [
        Notice(
            notice_id="NOTICE_ML_HYD",
            title="Machine Learning Intern",
            organization="DeepTech Solutions",
            raw_text="Join our AI lab working on computer vision and machine learning in Hyderabad. Strong Python required.",
            skills=["Python", "Machine Learning", "PyTorch"],
            degree="B.Tech",
            branches=["CSE", "ECE", "Data Science"],
            location="Hyderabad",
            mode="In-office",
            duration="6 months",
            stipend=30000.0,
            deadline=date(2026, 5, 1),
            deadline_status="active",
        ),
        Notice(
            notice_id="NOTICE_JAVA_PUNE",
            title="Java Backend Intern",
            organization="Enterprise Cloud Inc",
            raw_text="Seeking backend interns proficient in Java, Spring Boot, and SQL databases located in Pune.",
            skills=["Java", "Spring Boot", "SQL"],
            degree="B.Tech",
            branches=["CSE", "IT"],
            location="Pune",
            mode="Hybrid",
            duration="3 months",
            stipend=20000.0,
            deadline=date(2026, 4, 15),
            deadline_status="active",
        ),
        Notice(
            notice_id="NOTICE_REACT_BLR",
            title="Frontend Web Developer Intern",
            organization="Creative Apps Studio",
            raw_text="Exciting opportunity for frontend developers working on modern web UI with React, Next.js, and CSS.",
            skills=["React", "TypeScript", "Next.js"],
            degree="B.Tech",
            branches=["CSE", "IT", "ECE"],
            location="Bangalore",
            mode="Remote",
            duration="6 months",
            stipend=25000.0,
            deadline=date(2026, 6, 1),
            deadline_status="active",
        ),
        Notice(
            notice_id="NOTICE_EMBEDDED_DEL",
            title="Embedded Firmware Intern",
            organization="IoT Hardware Labs",
            raw_text="Hands-on role developing micro-controller C and C++ firmware in Delhi. Hardware debugging skills needed.",
            skills=["C", "C++", "Embedded Systems"],
            degree="B.Tech",
            branches=["ECE", "EEE"],
            location="Delhi",
            mode="In-office",
            duration="4 months",
            stipend=18000.0,
            deadline=date(2026, 4, 30),
            deadline_status="active",
        ),
    ]


def test_1_relevant_notice_ranked_above_unrelated(sample_notices: list[Notice]) -> None:
    """Test 1: Verify a notice containing Python, ML, and Hyderabad ranks above unrelated notices."""
    retriever = BM25Retriever()
    retriever.build(sample_notices)

    query = "Python machine learning Hyderabad"
    results = retriever.search(query, top_k=10)

    assert len(results) > 0
    top_notice_id, top_score = results[0]

    # The ML internship in Hyderabad must be ranked first with a positive score
    assert top_notice_id == "NOTICE_ML_HYD"
    assert top_score > 0.0

    # Score of top notice must be strictly higher than all other notices
    for nid, score in results[1:]:
        assert top_score > score


def test_2_top_k_limits(sample_notices: list[Notice]) -> None:
    """Test 2: Verify top_k parameter correctly constrains the number of returned results."""
    retriever = BM25Retriever()
    retriever.build(sample_notices)

    # top_k smaller than corpus size
    results_k2 = retriever.search("Python Java", top_k=2)
    assert len(results_k2) == 2

    # top_k equal to 1
    results_k1 = retriever.search("Python Java", top_k=1)
    assert len(results_k1) == 1

    # top_k greater than corpus size (4 items) returns at most corpus size without crashing
    results_k50 = retriever.search("Python Java", top_k=50)
    assert len(results_k50) == len(sample_notices)


def test_3_empty_and_whitespace_query_does_not_crash(sample_notices: list[Notice]) -> None:
    """Test 3: Verify empty query, whitespace query, or no-match tokens do not crash."""
    retriever = BM25Retriever()
    retriever.build(sample_notices)

    # Empty string
    results_empty = retriever.search("")
    assert results_empty == []

    # Whitespace string
    results_ws = retriever.search("   \t\n  ")
    assert results_ws == []

    # Punctuation only string
    results_punct = retriever.search("... ,,, !!!")
    assert results_punct == []


def test_4_notice_ids_correspond_to_original_notices(sample_notices: list[Notice]) -> None:
    """Test 4: Verify notice IDs returned by search correspond exactly to original notices."""
    retriever = BM25Retriever()
    retriever.build(sample_notices)

    valid_ids = {n.notice_id for n in sample_notices}
    results = retriever.search("Java Spring Boot", top_k=10)

    assert len(results) > 0
    for notice_id, score in results:
        assert notice_id in valid_ids
        assert isinstance(notice_id, str)
        assert isinstance(score, float)


def test_5_realistic_corpus_build_and_search() -> None:
    """Test 5: Verify build and search with a realistic multi-domain internship corpus."""
    realistic_corpus = [
        Notice(
            notice_id="N_DATA_ENG",
            title="Data Engineering Intern",
            organization="FinTech Analytics",
            raw_text="Building data pipelines using Apache Spark, Kafka, and Python in Mumbai. SQL optimization required.",
            skills=["Python", "Apache Spark", "Kafka", "SQL"],
            degree="B.Tech",
            branches=["CSE", "IT"],
            location="Mumbai",
            mode="Hybrid",
            duration="6 months",
        ),
        Notice(
            notice_id="N_CYBER_SEC",
            title="Cybersecurity Analyst Intern",
            organization="SecureNet Defense",
            raw_text="Conducting penetration testing, vulnerability assessment, and network monitoring in Gurgaon.",
            skills=["Network Security", "Linux", "Python", "Wireshark"],
            degree="B.Tech",
            branches=["CSE", "IT", "ECE"],
            location="Gurgaon",
            mode="In-office",
            duration="3 months",
        ),
        Notice(
            notice_id="N_DEVOPS",
            title="DevOps & Cloud Engineer Intern",
            organization="CloudScale Infra",
            raw_text="Managing Docker containers, Kubernetes clusters, and AWS infrastructure in Bangalore.",
            skills=["Docker", "Kubernetes", "AWS", "Terraform"],
            degree="B.Tech",
            branches=["CSE", "IT"],
            location="Bangalore",
            mode="Remote",
            duration="6 months",
        ),
    ]

    retriever = BM25Retriever()
    retriever.build(realistic_corpus)

    # Search for cloud and DevOps skills
    cloud_results = retriever.search("Kubernetes Docker AWS Bangalore", top_k=3)
    assert len(cloud_results) > 0
    assert cloud_results[0][0] == "N_DEVOPS"

    # Search for cybersecurity
    sec_results = retriever.search("penetration testing vulnerability Gurgaon", top_k=3)
    assert len(sec_results) > 0
    assert sec_results[0][0] == "N_CYBER_SEC"


def test_edge_cases_empty_corpus_and_invalid_top_k() -> None:
    """Additional edge cases: uninitialized index, empty corpus, and negative/zero top_k."""
    retriever = BM25Retriever()

    # Search on uninitialized retriever
    assert retriever.search("python") == []

    # Build on empty list
    retriever.build([])
    assert retriever.search("python") == []

    # top_k <= 0
    notices = [
        Notice(
            notice_id="N1",
            title="Test Title",
            raw_text="Test description with Python",
            skills=["Python"],
        )
    ]
    retriever.build(notices)
    assert retriever.search("python", top_k=0) == []
    assert retriever.search("python", top_k=-5) == []


def test_tokenizer_preserves_technical_terms() -> None:
    """Verify tokenizer retains C++, C#, .NET, Node.js, and version/hyphen terms."""
    text = "We need C++, C#, .NET, Node.js, and scikit-learn in B.Tech."
    tokens = tokenize(text)
    assert "c++" in tokens
    assert "c#" in tokens
    assert ".net" in tokens
    assert "node.js" in tokens
    assert "scikit-learn" in tokens
    assert "b.tech" in tokens
