
import us
import pandas as pd

# Set focus states.
focus = { us.states.FL, us.states.IL, us.states.MA, us.states.MD, us.states.TX }

# Get the configuration for the state.
summary = pd.read_csv("./minority-RCV/data/demographics/summary.csv")\
    .set_index("STATE")\
    .to_dict("index")

# Get everything else!
template = ""

for state in us.states.STATES:
    if state in focus: continue
    data = summary[state.name.title()]

    # Get the subsection header.
    template += f"\\subsubsection{{{state.name.title()}}}\n"

    # Create the caption.
    if data["REPRESENTATIVES"] > 5:
        caption = f"Neutral ensemble and simulated RCV election results for {state.name.title()}."
        template += f"\\combinedfigure{{{state.abbr}}}{{{caption}}}\n"
    else:
        caption = f"Simulated statewide RCV election results for {state.name.title()}."
        template += f"\\combinedfigurenochain{{{state.abbr}}}{{{caption}}}\n"

    template += "\\clearpage\n\n"

# Write to file.
with open("extra-states.tex", "w") as w: w.write(template)
