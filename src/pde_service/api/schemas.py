"""Request schemas. Pydantic rejects malformed requests (wrong types, missing fields,
out-of-range preset parameters) before they reach the model."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class SineTerm(BaseModel):
    amp: float = Field(ge=-3, le=3)
    k: int = Field(ge=1, le=64, description="integer wavenumber (periodic domain)")
    phase: float = Field(default=0.0, ge=-10, le=10)


class SinesIC(BaseModel):
    type: Literal["sines"]
    terms: list[SineTerm] = Field(min_length=1, max_length=8)


class GaussianIC(BaseModel):
    type: Literal["gaussian"]
    amp: float = Field(ge=-3, le=3)
    center: float = Field(ge=0, le=1)
    width: float = Field(gt=0.005, le=0.5)


class TestSampleIC(BaseModel):
    type: Literal["pdebench_test"]
    index: int = Field(ge=0)


InitialCondition = Annotated[SinesIC | GaussianIC | TestSampleIC, Field(discriminator="type")]


class ICRequest(BaseModel):
    nu: float = 0.1
    ic: InitialCondition


class PredictRequest(BaseModel):
    nu: float
    u0: list[float] | None = Field(default=None, max_length=4096)
    test_index: int | None = Field(default=None, ge=0, description="held-out PDEBench test sample")
    run_solver: bool = True

    @model_validator(mode="after")
    def exactly_one_input(self):
        if (self.u0 is None) == (self.test_index is None):
            raise ValueError("provide exactly one of 'u0' or 'test_index'")
        return self
