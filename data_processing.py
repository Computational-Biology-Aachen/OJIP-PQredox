# %%
import pandas as pd
import numpy as np
from pathlib import Path
import utils as utl
import matplotlib.pyplot as plt
from functools import partial
import re

from matplotlib import cm

pd.set_option('future.no_silent_downcasting', True)

idx = pd.IndexSlice

# Define default indexer for easier indexing
def didx(
    experiment: str | list[str | int] | slice=slice(None),
    strain: str | list[str | int] | slice=slice(None),
    condition: str | list[str | int] | slice=slice(None),
    treatment: str | list[str | int] | slice=slice(None),
    replicate: str | list[str | int] | slice=slice(None),
):
    res = idx[experiment, strain, condition, treatment, replicate]

    return tuple([x for x in res if x is not None])


# Import raw data
def load_data(FILEPATHS):
    # Create overall container for ojip data
    ojip = ind = None

    _ojips = _inds= [None] * len(FILEPATHS)

    for j, (FILEPATH, _ojip, _ind) in enumerate(zip(FILEPATHS, _ojips, _inds)):
        files = list(Path(FILEPATH).glob("*_*.*"))
        print(f"Importing: {FILEPATH}\t\t|  {len(files)} files")

        # Read in the sample ODs
        # ods = pd.read_excel(Path(FILEPATH.parent) / "ODs.xlsx", index_col=[0,1,2])

        # Read in the remeasured samples
        try:
            remeasured_samples = pd.read_excel(Path(FILEPATH.parent) / "remeasurements.xlsx", index_col=0)
        except Exception:
            remeasured_samples = None

        # Read in the conditions map
        try:
            conditions_map = pd.read_csv(Path(FILEPATH.parent) / "conditions_map.csv", index_col=0).iloc[:,0]
            conditions_map.index = conditions_map.index.astype(str)
        except Exception:
            conditions_map = None 

        

        # Read the OJIP files
        for i, file in enumerate(files):     
            # FL6000 measurement
            if FILEPATH.name.startswith("FL6000"):
                # Read file
                _df = pd.read_table(file, sep="\t", skiprows=17, header=None)
                _df = _df.set_index(0)

            # AquaPen measurement
            elif FILEPATH.name.startswith("AquaPen"):
                # Read file
                _df = pd.read_table(file, sep="\t", index_col=0, skiprows=8, header=None, skipfooter = 38, engine="python").iloc[:,:-1]

            elif FILEPATH.name.startswith("MC_PAM"):
                try:
                    _df = pd.read_csv(file, sep=";", index_col=0)[["Fluo, V"]]
                except UnicodeDecodeError as e:
                    print(f"File corrupted: {file}")
                except Exception as e:
                    raise RuntimeError(e)
                

            # Create a common, formatted dataframe
            try:
                _ojip, _ind = utl.add_and_format_df(
                    df=_df,
                    index=i,
                    full_df=_ojip,
                    ind=_ind,
                    files=files,
                    index_fields=["Strain", "CO2", "Experiment", "Treatment",  "Replicate"],
                    remeasured_df=remeasured_samples
                )
            except Exception as e:
                print(f"{file.name}\n{e}")
        # Drop missing columns
        _ojip = _ojip.dropna(axis=1)
        _ind = _ind.dropna(axis=1)

        # Set the correct elements and order of the index
        _ind = _ind.loc[["Experiment","Strain", "CO2", "Treatment", "Replicate"]]

        # Remove the prefixes from the columns
        if conditions_map is not None:
            _ind.loc["Treatment"] = _ind.loc["Treatment"].replace(conditions_map).astype(float)
        else:
            _sub = partial(re.sub, "^([0-9]+)[^0-9]*$", "\\1")
            _ind.loc["Treatment"] = _ind.loc["Treatment"].apply(_sub).astype(int)
        _ind.loc["Replicate"] = _ind.loc["Replicate"].str.removeprefix("Rep").astype(int)

        # Set the index
        _ojip.columns = pd.MultiIndex.from_frame(_ind.T)
        _ojip = _ojip.sort_index()

        if FILEPATH.name.startswith("FL6000"):
            _ojip.index = _ojip.index * 1e3
        elif FILEPATH.name.startswith("AquaPen"):
            _ojip.index = _ojip.index * 1e-3

        _ojip = _ojip.sort_index(axis=1)

        # Exclude early timepoints
        _ojip = _ojip.loc[1e-20:]

        _ojips[j]=_ojip

    # Collect all ojips
    ojip = pd.concat(_ojips, axis=1)
    ojip = ojip.sort_index()

    # # Get the strains, ods and light intensities
    # experiments = np.sort(ojip.columns.to_frame()["Experiment"].unique())
    # strains = np.sort(ojip.columns.to_frame()["Strain"].unique())

    # conditions = ojip.columns.to_frame().loc[:,"CO2"].groupby("Strain").unique().apply(np.sort)
    # treatments = ojip.columns.to_frame().loc[:,"Treatment"].groupby(["Strain"]).unique().apply(np.sort)
    # replicates = ojip.columns.to_frame().loc[:,"Replicate"].groupby(["Strain", "CO2"]).unique().apply(np.sort)

    # return ojip, experiments, strains, conditions, treatments, replicates

    # Get the strains, ods and light intensities
    levels = {
        "experiments": np.sort(ojip.columns.to_frame()["Experiment"].unique()),
        "strains": np.sort(ojip.columns.to_frame()["Strain"].unique()),
        "conditions": ojip.columns.to_frame().loc[:,"CO2"].groupby("Strain").unique().apply(np.sort),
        "treatments": ojip.columns.to_frame().loc[:,"Treatment"].groupby(["Strain"]).unique().apply(np.sort),
        "replicates": ojip.columns.to_frame().loc[:,"Replicate"].groupby(["Strain", "CO2"]).unique().apply(np.sort),
    }

    return ojip, levels

def get_ojip_features(ojip):
# %%
    ojip_features = pd.DataFrame(index=ojip.columns, columns=["F0", "FM", "Fv", "Fv/FM", "FJapprox", "VJapprox"])
    ojip_features = ojip_features.sort_index()

    # Determine F0
    ojip_features.loc[:, "F0"] = ojip.loc[:0.03, :].dropna().mean() # :0.05

    # Determine FM
    ojip_features.loc[:, "FM"] = ojip.dropna().rolling(5).mean().max()

    # Determine FM_timing
    ojip_features.loc[:, "FMtiming"] = ojip.dropna().rolling(5).mean().idxmax()

    # Determine Fv
    ojip_features.loc[:, "Fv"] = ojip_features["FM"] - ojip_features["F0"]

    # Calculate Fv/Fm
    ojip_features.loc[:, "Fv/FM"] = ojip_features["Fv"] / ojip_features["FM"]


    # Determine FJ approximately as the maximum
    ojip_features.loc[:, "FJapprox"] = ojip.loc[0.1:5, :].dropna().rolling(5).mean().max()

    # Determine FJ approximately as the maximum
    ojip_features.loc[:, "VJapprox"] = (ojip_features["FJapprox"] - ojip_features["F0"]) / ojip_features["Fv"]

    # Determine the mean and sd
    ojip_features_meansd = pd.concat({
        "mean":ojip_features.groupby(ojip_features.index.names[:-1]).mean(),
        "sd":ojip_features.groupby(ojip_features.index.names[:-1]).std()
        }, names=["Replicate"]).reorder_levels(list(ojip_features.index.names))
    
    return ojip_features, ojip_features_meansd


# %%
def normalize_ojip(ojip, ojip_features, ojip_features_meansd=None, norm_to_control = False):
    if norm_to_control:
        ojip_norm = ojip.copy()
        
        for strain in levels["strains"]:
            for condition in levels["conditions"][strain]:
                _ind = didx(strain=strain, condition=condition)
                
                # Subtract the dark control F0
                ojip_norm.loc[:, _ind] = ojip_norm.loc[:, _ind] - ojip_features_meansd.loc[didx(strain=strain, condition=condition, replicate="mean", treatment=0), "F0"].iloc[0]

                # Divide by the dark control FM
                ojip_norm.loc[:, _ind] = ojip_norm.loc[:, _ind] / (
                    ojip_features_meansd.loc[didx(strain=strain, condition=condition, replicate="mean", treatment=0), "FM"].iloc[0] -
                    ojip_features_meansd.loc[didx(strain=strain, condition=condition, replicate="mean", treatment=0), "F0"].iloc[0]
                )

    else:
        ojip_norm = ojip - ojip_features["F0"]
        ojip_norm = ojip_norm / (ojip_features["FM"] - ojip_features["F0"])

        ojip_norm = ojip_norm.astype(float)

    return ojip_norm

# Defule defualt options for finding options
feature_finding_options =     {
        "FJ_time_min" : 0.5, # 0.1  # (float(str(request.form.get('FJ_time_min')))) 
        "FJ_time_max" : 5, # 30   # (float(str(request.form.get('FJ_time_max')))) 
        "FI_time_min" : 5,   # (float(str(request.form.get('FI_time_min')))) 
        "FI_time_max" : 500,  # (float(str(request.form.get('FI_time_max')))) 
        "FP_time_min" : 40,  # (float(str(request.form.get('FP_time_min')))) 
        "FP_time_max" : 590, # (float(str(request.form.get('FP_time_max')))) 
    }


# Plot
def plot_ojip(ojip, levels, ojip_points=None, point_finding_span=None, treatment_label="TREATMENT UNIT", cmap=cm.cool):
    fig, axes = plt.subplots(
        2,
        2,
        sharex=True, 
        sharey=True,
        figsize=(10,10)
    )

    for s, strain in enumerate(levels["strains"]):
        # Create the colormap for the light intensity
        colornorm = plt.Normalize(
            levels["treatments"][strain].min(),
            levels["treatments"][strain].max()
        )

        for c ,condition in enumerate(levels["conditions"][strain]):
            ax = axes[s,c]

            try:
                dat = ojip.loc[:, didx(strain=strain, condition=condition,replicate=levels["replicates"][(strain,condition)],treatment=levels["treatments"][strain])].dropna()
            except KeyError:
                ax.grid(which="both")
                continue

            for nam, row in dat.T.iterrows():
                ax.plot(row, c=cmap(colornorm(nam[-2])))
            ax.set_title(f"{strain} - {condition}")
            ax.set_ylabel("Fluorescence")

            if ojip_points is not None:
                ax.plot(
                    ojip_points.loc[didx(strain=strain, condition=condition,replicate=levels["replicates"][(strain,condition)],treatment=levels["treatments"][strain]), ("grad2-min", "FJ_time")],
                    ojip_points.loc[didx(strain=strain, condition=condition,replicate=levels["replicates"][(strain,condition)],treatment=levels["treatments"][strain]), ("grad2-min", "FJ_value")],
                    marker="x",
                    ls="",
                    c="k"
                )

                ax.plot(
                    ojip_points.loc[didx(strain=strain, condition=condition,replicate=levels["replicates"][(strain,condition)],treatment=levels["treatments"][strain]), ("inflection", "FJ_time")],
                    ojip_points.loc[didx(strain=strain, condition=condition,replicate=levels["replicates"][(strain,condition)],treatment=levels["treatments"][strain]), ("inflection", "FJ_value")],
                    marker="o", fillstyle='none', c='k',
                    ls="",
                    markersize=4
                )

                ax.plot(
                    ojip_points.loc[didx(strain=strain, condition=condition,replicate=levels["replicates"][(strain,condition)],treatment=levels["treatments"][strain]), ("grad1-max", "FP_time")],
                    ojip_points.loc[didx(strain=strain, condition=condition,replicate=levels["replicates"][(strain,condition)],treatment=levels["treatments"][strain]), ("grad1-max", "FP_value")],
                    marker="^",
                    ls="",
                    c="k"
                )
            ax.axvline(1, c="k", ls="--")
            ax.grid(which="both")

            if point_finding_span is not None:
                ax.axvspan(*point_finding_span, color="grey", alpha=0.5)


    for ax in axes.flatten():
        ax.set_xscale("log")
        ax.set_xlabel("Time [ms]")
    # axes[0,-1].legend(loc="upper left", bbox_to_anchor=(1,1), title="light intensity")

    fig.tight_layout()
    fig.subplots_adjust(right=0.9)

    # Add a colorbar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    fig.colorbar(
        cm.ScalarMappable(norm=colornorm, cmap=cmap),
                cax=cax, orientation='vertical', label=treatment_label)

    # fig.savefig(f"figures/OJIP_per_OD_{strain}.png")
    return fig, axes


# %%
def plot_VJ_per_replicate(VJ, levels, title=None, treatment_label="TREATMENT UNIT",):
    fig, axes = plt.subplots(
        2, 
        2, 
        sharex=True, 
        # sharey="row",
        sharey=True,
        figsize=(10,10),
    )
    for i, strain in enumerate(levels["strains"]):
        for j, condition in enumerate(levels["conditions"][strain]):
            ax = axes[i,j]

            for replicate in levels["replicates"][(strain,condition)]:
                try:
                    dat = VJ.loc[didx(strain=strain, condition=condition,replicate=replicate,treatment=levels["treatments"][strain])].dropna()
                except KeyError:
                    ax.grid(which="both")
                    continue
                
                treatment_levels = dat.index.get_level_values("Treatment").to_numpy()

                ax.plot(
                    treatment_levels,
                    dat,
                    marker="o",
                    ls="",
                    label=replicate
                )
                ax.legend()
            ax.grid(which="both")

            ax.set_title(f"{strain} - {condition}")

    # Set the x labels
    for ax in axes[-1,:]:
        ax.set_xlabel(treatment_label)

    # Set the y labels
    for ax in axes[:, 0]:
        ax.set_ylabel("VJ [rel.]")

    if title is not None:
        fig.suptitle(title, size=20, weight='bold', y=1)

    for ax in axes.flatten():
        ax.axhline(0.7, c="k")

    fig.tight_layout()
    plt.show(fig)
    plt.close(fig)


# %%
def get_common_time_VJ(ojip_points, ojip_norm, levels, point_selector=("inflection", "FJ_time")):
    # Get the mean VJ timing of the dark adapted samples
    VJ_timing = ojip_points.loc[
        didx(treatment=0),
        point_selector
    ].groupby(ojip_points.index.names[:-1]).mean().droplevel(-1)

    # Get a time range 5% around the identified point
    VJ_timing_range = pd.DataFrame({
        "min": VJ_timing * 0.99,
        "max": VJ_timing * 1.01
    })

    # Get the VJ at the common time point
    VJ_values = pd.Series(index=ojip_points.index)

    for strain in levels["strains"]:
        for condition in levels["conditions"][strain]:
            _ind = didx(strain=strain, condition=condition)
            dat = ojip_norm.loc[
                VJ_timing_range.loc[_ind, "min"].iloc[0]:VJ_timing_range.loc[_ind, "max"].iloc[0]
                ,_ind]
            VJ_values.loc[_ind] = dat.mean()

    return VJ_timing, VJ_timing_range, VJ_values
