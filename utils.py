import numpy as np
import pandas as pd
from functools import partial
import warnings
idx = pd.IndexSlice
import warnings

import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib import colors
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

from data_processing import didx

def add_and_format_df(df, index, full_df, ind, files, index_fields, remeasured_df=None):
    file = files[index]

    if index == 0:
        # Create container for data and index
        full_df = pd.DataFrame(np.nan, index = df.index, columns = np.arange(len(files)))
        ind = pd.DataFrame(np.nan, index=index_fields, columns= np.arange(len(files)))

    # Make the measurements index
    
    sample_name = file.name.removesuffix(file.suffix)
    if remeasured_df is not None and sample_name in remeasured_df.index:
        # If the sample shouldnet be used, return the previous df and index
        if not remeasured_df.loc[sample_name, "use"]:
            print(f"{sample_name}: was remeasured, Skipping...")
            return full_df, ind
        else:
            new_name = remeasured_df.loc[sample_name, "New_name"]
            _ind = new_name.split("_")
    else:
        _ind = sample_name.split("_")
    
    ind[index]=_ind

    untracked_ind = [x for x in df.index if x not in full_df.index]
    if len(untracked_ind) > 0:
        full_df = pd.concat([
            full_df,
            pd.DataFrame(np.nan, index=untracked_ind, columns=full_df.columns)
        ])
    full_df[index] = df
    
    return full_df, ind


# Function to determine OJIP points using polynomial fits
def determine_OJIP_points(
        OJIP_double_normalized,
        knots_reduction_factor=10,
        FJ_time_min:float = 0.5, # 0.1  # (float(str(request.form.get('FJ_time_min')))) 
        FJ_time_max:float = 50,   # (float(str(request.form.get('FJ_time_max')))) 
        FJ_time_exp:float|None = None,   # (float(str(request.form.get('FJ_time_max')))) 
        FI_time_min:float = 10,   # (float(str(request.form.get('FI_time_min')))) 
        FI_time_max:float = 500,  # (float(str(request.form.get('FI_time_max')))) 
        FI_time_exp:float|None = None,  # (float(str(request.form.get('FI_time_max')))) 
        FP_time_min:float = 100,  # (float(str(request.form.get('FP_time_min')))) 
        FP_time_max:float = 1000, # (float(str(request.form.get('FP_time_max'))))
        FP_time_exp:float|None = None, # (float(str(request.form.get('FP_time_max'))))
        choose_method="extrema",
        polynom_degree=9,
        return_derivatives=False,
        return_fits=False,
        fitting_borders_extend_factor = 0.2
        ):

    # Make time logarithmic and without gaps (important for AquaPen)
    _time_raw = OJIP_double_normalized.index.to_numpy()
    time_raw = pd.Series(_time_raw, index=_time_raw)
    _time_log = np.geomspace(start=_time_raw[0], stop=_time_raw[-1], num=len(_time_raw))
    time_log = pd.Series(_time_log, index = _time_log) # type: ignore
    
    # Define the detected points
    detected_points = ["FJ", "FI", "FP"]

    # Get the ranges in which the points are detected
    detection_ranges = pd.DataFrame({
        "min": [FJ_time_min, FI_time_min, FP_time_min],
        "max": [FJ_time_max, FI_time_max, FP_time_max],
    }, index = detected_points)

    # Get the expected times
    expected_times = pd.Series([FJ_time_exp, FI_time_exp, FP_time_exp], index = detected_points)

    # Initialize containers
    _derivative_curves = {nam:pd.DataFrame(index = time_log.loc[_min:_max], columns=OJIP_double_normalized.columns) for nam, (_min, _max) in detection_ranges.iterrows()}
    derivative_curves = {nam:pd.concat({x:v.copy() for x in ["fitted", "derivative1", "derivative2", "derivative3", "derivative4"]}, axis=1) for nam,v in _derivative_curves.items()}

    _fitted_curves = {nam:pd.DataFrame(index = time_raw.loc[_min:_max], columns=OJIP_double_normalized.columns) for nam, (_min, _max) in detection_ranges.iterrows()}
    fitted_curves = {nam:pd.concat({x:v.copy() for x in ["raw","fitted", "residuals"]}, axis=1) for nam,v in _fitted_curves.items()}

    # _poly_coefs = {nam:pd.DataFrame(index = np.arange(polynom_degree), columns=OJIP_double_normalized.columns) for nam, (_min, _max) in detection_ranges.iterrows()}
    # poly_coefs = {nam:pd.concat({x:v.copy() for x in ["fitted", "derivative1", "derivative2", "derivative3", "derivative4"]}, axis=1) for nam,v in _fitted_curves.items()}

    # Create a container for the Times and values of FJ, FI and FP
    F_points_index = [(k, f"{i}_{j}") for i in ["FJ", "FI"] for j in ["value", "time"] for k in ["inflection", "grad2-min"]] + [(k, f"FP_{j}") for j in ["value", "time"] for k in ["grad1-max", "grad2-min"]] 

    F_points = pd.DataFrame(index=OJIP_double_normalized.columns, columns=pd.MultiIndex.from_tuples(F_points_index))

    all_poly_coefs = {}

    #############################
    # Fit polynomial curves
    #############################

    for F in detected_points:
        # Subset the data
        # Use a slightly larger subset for the fitting
        dat_fit = OJIP_double_normalized.loc[
            detection_ranges.loc[F, "min"] * (1-fitting_borders_extend_factor): 
            detection_ranges.loc[F, "max"] * (1+fitting_borders_extend_factor)
        ]

        _time_raw_fit = dat_fit.index.to_numpy()

        dat = OJIP_double_normalized.loc[detection_ranges.loc[F, "min"]: detection_ranges.loc[F, "max"]
        ]
        
        _time_raw = dat.index.to_numpy()

        # Populate the used polynomial fit function
        _polyfit = partial(np.polyfit, np.log(_time_raw_fit), deg=polynom_degree)

        # Get the polynomial coefficients
        poly_coefs = dat_fit.apply(_polyfit)
        all_poly_coefs[F] = poly_coefs

        for sample in poly_coefs.columns:
            # Create the polynomial model
            poly_model = np.poly1d(poly_coefs[sample])

            if return_fits:
                # Save the raw values
                fitted_curves[F].loc[:,idx["raw", *sample]] = dat.loc[:,sample].to_numpy()

                # Calculate the fitted values
                fitted_curves[F].loc[:,idx["fitted", *sample]] = poly_model(np.log(_time_raw))

                # Calculate the residuals
                fitted_curves[F].loc[:,idx["residuals", *sample]] = fitted_curves[F].loc[:,idx["fitted", *sample]] - fitted_curves[F].loc[:,idx["raw", *sample]]

            # Calculate the derivative polynomial coefficients
            if F in ["FI", "FJ"]:
                methods = ["inflection", "grad2-min"]
            else:
                methods = ["grad1-max", "grad2-min"]

            for method in methods:
                if method == "inflection":
                    deg_target = 1
                    deg_root = 2
                    deg_posneg = 3
                elif method == "grad2-min":
                    deg_target = 2
                    deg_root = 3
                    deg_posneg = 4
                else:
                    deg_target = 0
                    deg_root = 1
                    deg_posneg = -2

                # Get the coefficients for the derivative
                deriv_root_coef = np.polyder(poly_coefs[sample], deg_root)
                
                # Get the x-intercepts of the derivative
                deriv_roots = np.roots(deriv_root_coef)

                # Remove complex solutions
                deriv_roots = np.sort(deriv_roots.real[abs(deriv_roots.imag)<1e-5])

                # Only consider solutions in the detection range
                deriv_roots = deriv_roots[np.logical_and(
                    deriv_roots >= np.log(detection_ranges.loc[F, "min"]),
                    deriv_roots <= np.log(detection_ranges.loc[F, "max"]),
                )]

                # If no viable roots are found skip this sample
                if len(deriv_roots) == 0:
                    continue
                else:
                    # Otherwise check if the next derivative is positive at the x-intercepts
                    if deg_posneg >0:
                        deriv_pos_coef = np.polyder(poly_coefs[sample], deg_posneg)
                        deriv_pos = np.poly1d(deriv_pos_coef)(deriv_roots)
                        potential_extrema = deriv_roots[deriv_pos>0]
                    else:
                        deriv_pos_coef = np.polyder(poly_coefs[sample], -deg_posneg)
                        deriv_pos = np.poly1d(deriv_pos_coef)(deriv_roots)
                        potential_extrema = deriv_roots[deriv_pos<0]

                    # If no viable roots are left skip this sample
                    if len(potential_extrema) == 0:
                        continue
                    elif len(potential_extrema) == 1:
                        # If only one viable point is found, use it
                        F_points.loc[sample, (method, f"{F}_time")] = np.exp(float(potential_extrema[0]))
                        F_points.loc[sample, (method, f"{F}_value")] = poly_model(potential_extrema[0])
                    else:
                        # If multiple interceps are possible, choose one depending on a defined method    
                        # Select the minimum that has the lowest minimum in the target gradient                  
                        if choose_method == "extrema":
                            # Get the model for the target derivative
                            if deg_target >0 :
                                deriv_target_coef = np.polyder(poly_coefs[sample], deg_target)
                                deriv_target_model = np.poly1d(deriv_target_coef)
                                extrema_index = deriv_target_model(potential_extrema).argmin()
                            else:
                                deriv_target_coef = poly_coefs[sample]
                                deriv_target_model = np.poly1d(deriv_target_coef)
                                extrema_index = deriv_target_model(potential_extrema).argmax()

                            # Evaluate the target derivative model at all potential roos and choose the one with the lowest value
                            F_points.loc[sample, (method, f"{F}_time")] = np.exp(float(potential_extrema[extrema_index]))
                            F_points.loc[sample, (method, f"{F}_value")] = poly_model(potential_extrema[extrema_index])
                        elif choose_method == "closest":
                            exp_time = expected_times.loc[F]
                            if exp_time is None:
                                raise ValueError("When using choose_method 'closest', please provide an expected time for all points.")
                            else:
                                closest_index = np.argmin(np.abs(potential_extrema - np.log(exp_time)))
                                F_points.loc[sample, (method, f"{F}_time")] = np.exp(float(potential_extrema[closest_index]))
                                F_points.loc[sample, (method, f"{F}_value")] = poly_model(potential_extrema[closest_index])
                        
                        else:
                            raise NotImplementedError(f"choose method {choose_method} not implemented")

        # Calculate and evaluate all derivatives if they should be returned
        if return_derivatives:
            # Get the timepoints to and evaluate the polynomial model
            _time_log = time_log.loc[detection_ranges.loc[F, "min"]: detection_ranges.loc[F, "max"]]

            # print(_time_raw)
            # print(_time_log)
            # raise Error("Stop")
            # Evaluate the polynomails and its degrees for all samples
            for sample in OJIP_double_normalized.columns:
                # Create the polynomial model
                poly_model = np.poly1d(poly_coefs[sample])
                derivative_curves[F].loc[:,idx[f"fitted", *sample]] = poly_model(np.log(_time_log))

                # Iterate through all degrees and evaluate the derived polynomial models
                for deg in [1,2,3,4]:
                    deriv_coef = np.polyder(poly_coefs[sample], m=deg)

                    # Get the coefficients for the derivative model
                    deriv_model = np.poly1d(deriv_coef)
                    derivative_curves[F].loc[:,idx[f"derivative{deg}", *sample]] = deriv_model(np.log(_time_log))

    # Create the return object
    res = {"points":F_points, "polynom_coefs":all_poly_coefs}

    if return_derivatives:
        # Add the derivatives to the return
        res["derivatives"] = derivative_curves
    
    if return_fits:
        res["fits"] = fitted_curves

        # # Test the reconstruction with log-scaled time
        # raw_curves_model_polycoef = np.polyfit(np.log(fitted_time), test_values_reduced, reconstructed_poly_deg)
        # raw_curves_model = np.poly1d(raw_curves_model_polycoef)
        # raw_curves_reconstructed = pd.DataFrame(raw_curves_model(np.log(time_axis_logarthmic_reduced.iloc[:, 0])))
        # # raw_curves_reconstructed = pd.DataFrame(raw_curves_model(np.log(time_axis_logarthmic.iloc[:, 0])))
        # raw_curves_reconstructed.index = time_axis_logarthmic_reduced.iloc[:, 0]

    return res

def _plot_normalized_OJIP(
    fig,
    ax,
    strain,
    condition,
    levels,
    ojip_norm,
    ojip_norm_meansd,
    ojip_points,
    VJ_timing,
    colornorm,
    conditions_map,
    treatment_label="TREATMENT UNIT",
    treatment_format="{treatment}",
    treatment_0_label=None,
    treatments_select=None,
    treatment_center=None,
    use_colorbar = False,
    cmap = cm.coolwarm,
    variance_sleeve_alpha=0.6,
    point_x_selector= ("inflection", "FJ_time"),
    point_y_selector= ("inflection", "FJ_value"),
    point_label="Inflection point",
    plot_replicates=False,
    legend=True,
    title=True,
    VJ_text_loc=(0.8, 0),
):
    for k, treatment in enumerate(levels["treatments"][strain]):

        if not use_colorbar:
            if treatment not in treatments_select:
                continue

        # Plot the double normalized ojip curves
        if plot_replicates:
            for r, replicate in enumerate(levels["replicates"][(strain,condition)]):
                try:
                    dat = ojip_norm.loc[:, didx(strain=strain, condition=condition,replicate=replicate,treatment=treatment)].dropna()
                except KeyError:
                    ax.grid(which="both")
                    continue

                ax.plot(
                    dat,
                    ls="-",
                    label=treatment if r==0 and use_colorbar else None,
                    c=cmap(colornorm(treatment)) if k!=0 else "k",
                    alpha=0.5
                )
        else:
            try:
                dat = ojip_norm_meansd["mean"].loc[:, didx(strain=strain, condition=condition,replicate=None,treatment=treatment)].dropna()
            except KeyError:
                ax.grid(which="both")
                continue

            ax.plot(
                dat,
                ls="-",
                label=(treatment_format.format(treatment=treatment) if treatment!=0 or treatment_0_label is None else treatment_0_label) if not use_colorbar else None,
                c=cmap(colornorm(treatment)) if k!=0 else "k",
                # zorder=1
            )

            dat_sd = ojip_norm_meansd["sd"].loc[:, didx(strain=strain, condition=condition,replicate=None,treatment=treatment)].dropna()

            ax.fill_between(
                dat.index.to_numpy(),
                (dat-dat_sd).to_numpy().flatten(),
                (dat+dat_sd).to_numpy().flatten(),
                ls="-",
                fc=cmap(colornorm(treatment)) if k!=0 else "k",
                alpha=variance_sleeve_alpha,
                # zorder=1
            )


        # Add the timing of the inflection point
        if ojip_points is not None:
            ojip_points_time= ojip_points.loc[didx(strain=strain, condition=condition,replicate=levels["replicates"][(strain,condition)],treatment=treatment), point_x_selector].mean()
            ojip_points_value = ojip_points.loc[didx(strain=strain, condition=condition,replicate=levels["replicates"][(strain,condition)],treatment=treatment), point_y_selector].mean()

            ax.plot(
                ojip_points_time,
                ojip_points_value,
                marker="x",
                ls="",
                c="k",
                label=point_label if treatment == treatments_select[-1] else None
            )


    ax.grid(which="both")
    if title:
        ax.set_title(f"{conditions_map[condition]}", loc="left", weight='bold', size=15)

    # Mark the VJ point timing
    if VJ_timing is not None:
        VJ_time = VJ_timing.loc[:, strain, condition].iloc[0]
        ax.axvline(VJ_time, c="k", ls="--")
        ax.text(VJ_time*VJ_text_loc[0], VJ_text_loc[1], f"V$_{{J}}$ ({VJ_time:.2f} ms)", va="top", ha="right")

    if legend:
        if use_colorbar:
            fig.colorbar(
        cm.ScalarMappable(norm=colornorm, cmap=cmap),
                ax=ax, orientation='vertical', label=treatment_label)
            ax.legend(loc="upper left") #, prop={'size': 9})
        else:
            # ax.legend(loc=(0.6,0.2)) #, prop={'size': 9})
            ax.legend(loc="lower right") #, prop={'size': 9})

    return fig, ax

# Set the x labels
def log_tick_formatter(val, pos):
    return f"{val:g}"  # Uses general format, removes scientific notation

# Get the base plot for papers
def get_base_plot(
    ojip_norm,
    ojip_norm_meansd,
    ojip_points,
    VJ_timing,
    VJ_values,
    VJ_values_meansd,
    levels,
    treatment_label="TREATMENT UNIT",
    treatment_format="{treatment}",
    treatment_0_label=None,
    treatments_select=None,
    treatment_center=None,
    plot_replicates = False,
    use_colorbar = False,
    mark_sampled = True,
    cmap = cm.coolwarm,
    plot_strains = ["Syn", "Chlo"],
    row_label_ys = [1, 0.49],
    right_column_y_label="V$_{J}$ (r.u.)",
    left_column_y_label = "Double normalized Fluorescence (r.u.)",
    right_column_mark_zero=False,
    strain_map = {
        "Syn": r'$\mathit{\boldsymbol{Synechocystis}}$',
        "Chlo": r'$\mathit{\boldsymbol{Chlorella}}$'
    },
    light_phases=None,
    broken_logx_firstvalue=False,
    right_column_legend_loc="lower right",
    variance_sleeve_alpha=0.6,
    point_x_selector= ("inflection", "FJ_time"),
    point_y_selector= ("inflection", "FJ_value"),
    point_label="Inflection point",
    return_subplot_arguments=False,
    ojip_ymin=None,
    ojip_ymax=None,
    feature_ymin=None,
    feature_ymax=None,
):

    fig, axes = plt.subplots(
        2, 
        3, 
        # sharex="col", 
        sharey="col",
        figsize=(15,10),
    )

    # cmap = cm.cool

    if use_colorbar or treatment_center is None:
        colornorm = plt.Normalize(
            levels["treatments"].apply(np.min).min(),
            levels["treatments"].apply(np.max).max()
        )
    else:
        colornorm = colors.TwoSlopeNorm(
            vmin=levels["treatments"].apply(np.min).min(),
            vcenter=treatment_center,
            vmax=levels["treatments"].apply(np.max).max()
        )

    # Define the order of the plots
    plot_conditions = ["lowCO2", "highCO2"]
    plot_conditions_colors = {
        "highCO2": np.array((0,176,80,255))/255,
        "lowCO2": np.array((146,208,80,255))/255
    }

    # Map the names of the conditions
    conditions_map = {
        "lowCO2": "Air",
        "highCO2": "High CO$_{2}$",
    }

    markers = {
        "lowCO2": "^",
        "highCO2": "o",
    }

    # If no selected treatments were given, use all
    if treatments_select is None:
        treatments_select = np.sort(np.unique(np.concatenate(levels["treatments"].values)))


    # Plot the double normalized ojip curves
    for i, strain in enumerate(plot_strains):
        for j, condition in enumerate([c for c in plot_conditions if c in levels["conditions"][strain]]):
            ax = axes[i,j]

            fig, ax = _plot_normalized_OJIP(
                fig=fig,
                ax=ax,
                strain=strain,
                condition=condition,
                levels=levels,
                ojip_norm=ojip_norm,
                ojip_norm_meansd=ojip_norm_meansd,
                ojip_points=ojip_points,
                VJ_timing=VJ_timing,
                colornorm=colornorm,#
                conditions_map=conditions_map,
                treatment_label=treatment_label,
                treatment_format=treatment_format,
                treatment_0_label=treatment_0_label,
                treatments_select=treatments_select,
                treatment_center=treatment_center,
                use_colorbar = use_colorbar,
                cmap = cmap,
                variance_sleeve_alpha=variance_sleeve_alpha,
                point_x_selector= point_x_selector,
                point_y_selector= point_y_selector,
                point_label=point_label,
                plot_replicates=plot_replicates,
            )

            # Set the given ylims
            if ojip_ymin is not None or ojip_ymax is not None:
                ax.set_ylim(ojip_ymin, ojip_ymax)

            # Plot the VJ value
            ax = axes[i,-1]

            if broken_logx_firstvalue:
                treatments = levels["treatments"][strain][1:]
            else:
                treatments = levels["treatments"][strain]

            if plot_replicates:
                for r, replicate in enumerate(levels["replicates"][(strain,condition)]):
                    try:
                        dat = VJ_values.loc[didx(strain=strain, condition=condition,replicate=replicate,treatment=treatments)].dropna()
                    except KeyError:
                        ax.grid(which="both")
                        continue
                    
                    treatment_levels = dat.index.get_level_values("Treatment").to_numpy()

                    ax.plot(
                        treatment_levels,
                        dat,
                        marker=markers[condition],
                        ls="-",
                        label=condition if r==0 else None,
                        c = plot_conditions_colors[condition]
                    )
            
            else:
                try:
                    dat = VJ_values_meansd["mean"].loc[didx(strain=strain, condition=condition,replicate=None,treatment=treatments)].dropna().droplevel([0,1,2])
                except KeyError:
                    ax.grid(which="both")
                    continue
                
                treatment_levels = dat.index.get_level_values("Treatment").to_numpy()

                # ax.plot(
                #     treatment_levels,
                #     dat,
                #     marker="o",
                #     ls="-",
                #     label=condition,
                #     c = plot_conditions_colors[condition]
                # )

                dat_sd = VJ_values_meansd["sd"].loc[didx(strain=strain, condition=condition,replicate=None,treatment=treatments)].dropna().droplevel([0,1,2])

                ax.errorbar(
                    dat.index.to_numpy(),
                    dat,
                    yerr=dat_sd.to_numpy().flatten(),
                    marker=markers[condition],
                    ls="-",
                    label=conditions_map[condition],
                    c = plot_conditions_colors[condition],
                    markeredgecolor="k",
                    ecolor='k',
                    capsize=3
                )

                if not use_colorbar and mark_sampled:
                    for k, treatment in enumerate(treatments):

                        if treatment not in treatments_select:
                            continue
                        
                        ax.errorbar(
                            treatment,
                            dat.loc[treatment],
                            yerr=dat_sd.loc[treatment],
                            marker=markers[condition],
                            ls="",
                            c = plot_conditions_colors[condition],
                            markeredgecolor="k",
                            markeredgewidth=2,
                            # markersize=8,
                            ecolor='k',
                        )

            ax.legend(loc=right_column_legend_loc)# prop={'size': 9})
            ax.grid(which="both")

            if right_column_mark_zero:
                ax.axhline(0, c="k", ls="--")

    for ax in axes[:,-1]:
        ax.grid(True)
        if not broken_logx_firstvalue:
            ax.set_ylabel(right_column_y_label)
        else:
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))

        ax.set_xlabel(treatment_label)

        # Set the given ylims
        if feature_ymin is not None or feature_ymax is not None:
            ax.set_ylim(feature_ymin, feature_ymax)

    for ax in axes[:,:-1].flatten():
        ax.set_xlabel("Time (ms)")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))

    # Set the y labels
    for ax in axes[:, :2].flatten():
        ax.set_ylabel(left_column_y_label)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)

    # Add a broken log x-axis
    if broken_logx_firstvalue:

        # Plot the double normalized ojip curves
        for i, strain in enumerate(plot_strains):
            for j, condition in enumerate([c for c in plot_conditions if c in levels["conditions"][strain]]):

                ax = axes[i,-1]
                # Plot the VJ value
                if j==0:
                    ax.set_xscale("log")

                    # Create a ne axis for the left side of the broken axis
                    ax.set_xlim(ax.get_xlim()[0]*0.5)

                    # Alternatively: Add an axis to the top to draw in
                    pos = ax.get_position()
                    ax.set_position([pos.x0+0.03, pos.y0, pos.width-0.03, pos.height])  # Increase height by 10%

                    ax.spines['left'].set_visible(False)
                    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                    # ax.grid(visible=True, which="both", axis="x")
                    # ax.yaxis.set_visible(False)
                    
                    brax = fig.add_axes([pos.x0, pos.y0, 0.02, pos.height])
                    brax.sharey(ax)
                    brax.spines['right'].set_visible(False)

                    # Add broken indicator
                    size_factor = ax.get_position().width / brax.get_position().width

                    d = .015  # how big to make the diagonal lines in axes coordinates
                    d_adj = d*size_factor
                    # arguments to pass plot, just so we don't keep repeating them
                    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
                    ax.plot((-d, +d), (1-d, 1+d), **kwargs)
                    ax.plot((-d, +d), (-d, +d), **kwargs)

                    kwargs.update(transform=brax.transAxes)  # switch to the bottom axes
                    brax.plot((1-d_adj, 1+d_adj), (-d, +d), **kwargs)
                    brax.plot((1-d_adj, 1+d_adj), (1-d, 1+d), **kwargs)

                if plot_replicates:
                    for r, replicate in enumerate(levels["replicates"][(strain,condition)]):
                        try:
                            dat = VJ_values.loc[didx(strain=strain, condition=condition,replicate=replicate,treatment=[levels["treatments"][strain][0]])].dropna()
                        except KeyError:
                            brax.grid(which="both", axis="both")
                            continue
                        
                        treatment_levels = dat.index.get_level_values("Treatment").to_numpy()

                        brax.plot(
                            treatment_levels,
                            dat,
                            marker=markers[condition],
                            ls="-",
                            label=condition if r==0 else None,
                            c = plot_conditions_colors[condition]
                        )
                        brax.grid(which="both", axis="both")

                
                else:
                    try:
                        dat = VJ_values_meansd["mean"].loc[didx(strain=strain, condition=condition,replicate=None,treatment=[levels["treatments"][strain][0]])].dropna().droplevel([0,1,2])
                    except KeyError:
                        brax.grid(which="both", axis="both")
                        continue
                    
                    treatment_levels = dat.index.get_level_values("Treatment").to_numpy()

                    dat_sd = VJ_values_meansd["sd"].loc[didx(strain=strain, condition=condition,replicate=None,treatment=[levels["treatments"][strain][0]])].dropna().droplevel([0,1,2])

                    brax.errorbar(
                        dat.index.to_numpy(),
                        dat,
                        yerr=dat_sd.to_numpy().flatten(),
                        marker=markers[condition],
                        ls="-",
                        label=conditions_map[condition],
                        c = plot_conditions_colors[condition],
                        markeredgecolor="k",
                        ecolor='k',
                        capsize=3
                    )

                    if not use_colorbar and mark_sampled:
                        treatment = levels["treatments"][strain][0]
                        if treatment in treatments_select:
                        
                            brax.errorbar(
                                treatment,
                                dat.loc[treatment],
                                yerr=dat_sd.loc[treatment],
                                marker=markers[condition],
                                ls="",
                                c = plot_conditions_colors[condition],
                                markeredgecolor="k",
                                markeredgewidth=2,
                                # markersize=8,
                                ecolor='k',
                            )
                    

                # brax.set_xlim(-levels["treatments"][strain][1]*0.5, levels["treatments"][strain][1]*0.5)
                brax.set_xticks([0])
                brax.set_ylabel(right_column_y_label)
                brax.grid(visible=True, which="both", axis="both")


                if right_column_mark_zero:
                    brax.axhline(0, c="k", ls="--")

    # Add row labels
    for y, strain in zip(row_label_ys ,plot_strains):
        fig.text(0.045,y, strain_map[strain], weight='bold', size=20)

    # Add figure labels
    for i, ax in enumerate(axes.flatten()):
        # y= 0.1 if (i+1)%3==0 else 0.9
        # ax.text(y,0.1,chr(65+i), transform=ax.transAxes, weight='bold', size=20)
        x = 0.07 if not (broken_logx_firstvalue and (i+1)%3==0) else -0.03
        ax.text(x,0.9,chr(65+i), transform=ax.transAxes, weight='bold', size=20)

    if light_phases is not None:
        for ax in axes[:, -1]:
            ax.set_xlim(-25)

            # Alternatively: Add an axis to the top to draw in
            pos = ax.get_position()
            # ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 1.1])  # Increase height by 10%
            lbax = fig.add_axes([pos.x0, pos.y0+pos.height, pos.width, 0.02])
            lbax.sharex(ax)

            lbax.xaxis.set_visible(False)
            lbax.yaxis.set_visible(False)

            # Annotate the light phases
            for _, (start, end, light) in light_phases.iterrows():
                add_light_annotation_rectangle(lbax,start,end,light)

    if not return_subplot_arguments: 
        return fig, axes
    else:
        res = {
                "levels":levels,
                "ojip_norm":ojip_norm,
                "ojip_norm_meansd":ojip_norm_meansd,
                "ojip_points":ojip_points,
                "VJ_timing":VJ_timing,
                "colornorm":colornorm,
                "conditions_map":conditions_map,
                "treatment_label":treatment_label,
                "treatment_format":treatment_format,
                "treatment_0_label":treatment_0_label,
                "treatments_select":treatments_select,
                "treatment_center":treatment_center,
                "use_colorbar": use_colorbar,
                "cmap": cmap,
                "variance_sleeve_alpha":variance_sleeve_alpha,
                "point_x_selector": point_x_selector,
                "point_y_selector": point_y_selector,
                "point_label":point_label,
                "plot_replicates":plot_replicates,
        }
        return fig, axes, res

lightbarmap ={
    "dark":{"facecolor":"k", "label":"Dark", "labelcolor":"white"},
    "white":{"facecolor":"w", "label":"Light"},
    "high-white":{"facecolor":"w", "label":"High-light", "hatch":"..", "alpha":0.35,             "hatch_linewidth":0.1},
    "FR":{"facecolor":(0.5,0,0), "label":"FR", "labelcolor":"white"},
}

def add_light_annotation_rectangle(lbax, start, end, light):
    rect_param = ((start,0), end-start, 1)

    # Add an empty rectangle below as basis without alpha
    lbax.add_patch(
        Rectangle(
            *rect_param,
            fill=False,
            edgecolor="black",
            in_layout=True,
            alpha=1,
        )
    )

    # Add the correct filling on top with alpha
    lbax.add_patch(
        Rectangle(
            *rect_param,
            fill=True,
            edgecolor=None,
            # in_layout=True,
            facecolor=lightbarmap[light].get("facecolor"),
            hatch=lightbarmap[light].get("hatch"),
            alpha=lightbarmap[light].get("alpha"),
            hatch_linewidth=lightbarmap[light].get("hatch_linewidth"),
        )
    )

    # Add the text annotation
    lbax.text(
        start + (end-start)/2,
        0.5,
        lightbarmap[light]["label"],
        va="center",
        ha="center",
        weight="bold",
        size=7,
        color=lightbarmap[light].get("labelcolor"),
        bbox=dict(facecolor=lightbarmap[light].get("facecolor"), alpha=0.5, edgecolor='none', pad=1)
        )
    
# Add an arrow to a plot to mark the application of a compound
def add_application_arrow(ax, x=0, offset=0, arrow_frac_len=0.1):
    ys = [lin.get_ydata()[0] for lin in ax.lines if lin.get_xdata()[0]==0]
    y = np.max(ys) + offset

    y_len=np.diff(ax.get_ylim()) * arrow_frac_len

    ax.annotate("", xytext=(2, y+y_len), xy=(2, y),
            arrowprops=dict(width=2, headwidth=7, headlength=7, facecolor='black'))