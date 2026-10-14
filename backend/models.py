from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


Difficulty = Literal["basic", "intermediate", "advanced"]


class InterviewConfig(BaseModel):
    role: str = Field(min_length=2, max_length=120)
    description: str = Field(default="", max_length=1500)
    skills: list[str] = Field(default_factory=list, max_length=10)
    starting_difficulty: Difficulty = "basic"
    question_count: int = Field(default=5, ge=3, le=15)
    technical_weight: int = Field(default=50, ge=0, le=100)
    communication_weight: int = Field(default=30, ge=0, le=100)
    evidence_weight: int = Field(default=20, ge=0, le=100)
    custom_questions: list[str] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def validate_configuration(self):
        if self.technical_weight + self.communication_weight + self.evidence_weight != 100:
            raise ValueError("Scoring weights must add up to 100")
        self.skills = [skill.strip() for skill in self.skills if skill.strip()]
        self.custom_questions = [question.strip() for question in self.custom_questions if question.strip()]
        if any(len(skill) > 80 for skill in self.skills):
            raise ValueError("Each skill must be 80 characters or fewer")
        if any(len(question) > 500 for question in self.custom_questions):
            raise ValueError("Each custom question must be 500 characters or fewer")
        return self


class CVAnalysis(BaseModel):
    summary: str
    technical_skills: dict[str, int]
    years_experience: str
    key_achievements: list[str]
    areas_for_improvement: list[str]


class Question(BaseModel):
    text: str = Field(min_length=5)
    topic: str
    kind: Literal["opening", "followup", "coverage", "custom"]
    difficulty: Difficulty


class Evaluation(BaseModel):
    technical_accuracy: float = Field(ge=0, le=10)
    communication: float = Field(ge=0, le=10)
    evidence: float = Field(ge=0, le=10)
    feedback: str
    adjust_difficulty: Literal["easier", "harder", "maintain"] = "maintain"
    followup_worthwhile: bool = False
    suggested_followup: str = ""


class CandidateReport(BaseModel):
    summary: str
    strengths: list[str]
    improvements: list[str]
    next_steps: list[str]


class RecruiterReport(BaseModel):
    summary: str
    strengths: list[str]
    concerns: list[str]
    evidence: list[str]


class ReportNarrative(BaseModel):
    candidate: CandidateReport
    recruiter: RecruiterReport


class AnswerRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=12000)
