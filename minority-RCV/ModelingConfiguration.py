
from pydantic import BaseModel, validator, root_validator
from typing import Optional
import math

class ModelingConfiguration(BaseModel):
    pp: float
    """
    POC support for POC candidates of choice.
    """

    pw: float
    """
    POC support for white candidates of choice.
    """

    ww: float
    """
    White support for white candidates of choice.
    """

    wp: float
    """
    White support for POC candidates of choice.
    """

    pocshare: float
    """
    The share of the voting-age population that are people of color.
    """

    seats: int
    """
    The number of available seats.
    """

    multiplier: Optional[float] = 1
    """
    Multiplier designating how many candidates are competing for the available
    seats.
    """

    poc: Optional[float]=1/2
    """
    Fraction of candidates who are POC.
    """

    pool: str
    """
    The voting pool.
    """

    ballots: Optional[int] = 20000
    """
    The number of ballots in the simulated election.
    """

    simulations: Optional[int] = 10
    """
    The number of simulated elections conducted.
    """

    model: str
    """
    The desired model.
    """

    concentration: list
    """
    The dictionary of concentration parameters which parameterize the n- or
    m-dimensional Dirichlet distribution from which support vectors are drawn.
    Defaults to the four typical scenarios outlined in the RCV paper.
    """

    concentrationname: str
    """
    Name of the concentration we're using in this experiment.
    """

    turnout: Optional[float] = 1
    """
    Specify the level of POC turnout. A float in [0,1] which is multiplied by
    `pocshare` to simulate lower-turnout elections.
    """

    @validator("model")
    def _validate_model(cls, model):
        allowable = {"plackett-luce", "bradley-terry", "crossover", "cambridge"}

        if model not in allowable:
            raise ValueError(f"Model name {model} invalid.")

        return model

    @validator("pocshare")
    def _validate_pocshare(cls, pocshare):
        assert pocshare <= 1
        return pocshare

    @root_validator
    def _validate_poc_polarization(cls, v):
        # Assert that the polarization parameters add up.
        assert math.isclose(v.get("pp") + v.get("pw"), 1)
        assert math.isclose(v.get("ww") + v.get("wp"), 1)

        # Create a parameter that says how many POC candidates there are, and
        # how many candidates in total.
        v["candidates"] = math.ceil(v.get("seats")*v.get("multiplier"))
        v["poccandidates"] = math.ceil(v.get("candidates")*v.get("poc"))
        v["wcandidates"] = v.get("candidates")-v.get("poccandidates")
        assert v.get("wcandidates") + v.get("poccandidates") == v.get("candidates")

        # Also compute turnout.
        assert 0 <= v.get("turnout") <= 1
        v["pocshare"] = v.get("pocshare")*v.get("turnout")
        
        return v
