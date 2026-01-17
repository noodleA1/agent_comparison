"""Pydantic models for course generation data structures."""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class LessonPlan(BaseModel):
    """A single lesson plan within a course."""
    lesson_number: int = Field(..., description="Lesson number (1-10)")
    title: str = Field(..., description="Lesson title")
    objectives: List[str] = Field(default_factory=list, description="Learning objectives")
    content_outline: List[str] = Field(default_factory=list, description="Main content points")
    activities: List[str] = Field(default_factory=list, description="Practical activities")
    resources: List[str] = Field(default_factory=list, description="Additional resources")
    citations: List[str] = Field(default_factory=list, description="Source citations with URLs")


class Syllabus(BaseModel):
    """Course syllabus with all lessons."""
    course_title: str = Field(..., description="Title of the course")
    course_objective: str = Field(..., description="Main learning objective")
    target_audience: str = Field(default="General learners", description="Target audience")
    difficulty_level: str = Field(default="Intermediate", description="Difficulty level")
    lessons: List[LessonPlan] = Field(default_factory=list, description="List of lesson plans")


class CoursePackage(BaseModel):
    """Complete course package with all materials."""
    syllabus: Syllabus
    research_sources: List[str] = Field(default_factory=list, description="Sources used in research")
    generation_metadata: dict = Field(default_factory=dict, description="Generation metadata")


class GenerationMetrics(BaseModel):
    """Metrics for tracking generation performance."""
    framework: str = Field(..., description="Framework name")
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_calls: int = 0
    jina_calls: int = 0
    errors: List[str] = Field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


class QualityScore(BaseModel):
    """Quality assessment of generated content."""
    score: float = Field(..., ge=0.0, le=1.0, description="Quality score 0.0-1.0")
    feedback: str = Field(default="", description="Qualitative feedback")
    issues: List[str] = Field(default_factory=list, description="Specific issues found")
    iteration: int = Field(default=1, description="Which refinement iteration")


class GapAssessment(BaseModel):
    """Assessment of gaps in course content from student perspective."""
    gaps_found: List[str] = Field(default_factory=list, description="Content gaps identified")
    missing_prerequisites: List[str] = Field(default_factory=list, description="Prerequisites not covered")
    unclear_concepts: List[str] = Field(default_factory=list, description="Concepts needing clarification")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    ready_for_publication: bool = Field(default=False, description="Whether course is ready")


class CostBreakdown(BaseModel):
    """Cost tracking for generation phases."""
    research_cost: float = Field(default=0.0, description="Cost of research phase")
    syllabus_cost: float = Field(default=0.0, description="Cost of syllabus generation")
    quality_loop_cost: float = Field(default=0.0, description="Cost of quality refinement")
    lesson_generation_cost: float = Field(default=0.0, description="Cost of lesson generation")
    gap_assessment_cost: float = Field(default=0.0, description="Cost of gap assessment")
    gap_refinement_cost: float = Field(default=0.0, description="Cost of gap-driven refinement")  # NEW
    total_cost: float = Field(default=0.0, description="Total cost")
    total_tokens: int = Field(default=0, description="Total tokens used")

    def calculate_total(self):
        """Recalculate total from components."""
        self.total_cost = (
            self.research_cost +
            self.syllabus_cost +
            self.quality_loop_cost +
            self.lesson_generation_cost +
            self.gap_assessment_cost +
            self.gap_refinement_cost  # NEW
        )


class EnhancedCoursePackage(BaseModel):
    """Enhanced course package with quality and gap assessment."""
    syllabus: Syllabus
    quality_score: Optional[QualityScore] = None
    gap_assessment: Optional[GapAssessment] = None
    cost_breakdown: Optional[CostBreakdown] = None
    research_sources: List[str] = Field(default_factory=list, description="Sources used")
    generation_metadata: dict = Field(default_factory=dict, description="Metadata")


class FrameworkResult(BaseModel):
    """Result from a single framework's course generation."""
    framework: str
    success: bool
    course: Optional[CoursePackage] = None
    enhanced_course: Optional[EnhancedCoursePackage] = None
    metrics: Optional[GenerationMetrics] = None
    error: Optional[str] = None
    console_output: List[str] = Field(default_factory=list)
