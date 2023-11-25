# packages
import numpy as np 
import yaml, os, logging, sys
import pandas as pd
pd.set_option("display.max_columns", None)
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.dates import DayLocator, MonthLocator, DateFormatter
from scipy import optimize
from matplotlib.colors import LogNorm

from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# location of the scripts
sys.path.insert(0, '/fefs/aswg/workspace/juan.jimenez/stereo_analysis/scripts')
import auxiliar as aux
import geometry as geom
aux.params()

import itertools
from astropy import units as u
from astropy.constants import c
from astropy.coordinates import SkyCoord
from astropy.time import Time
from regions import CircleSkyRegion, PointSkyRegion
from datetime import datetime, timedelta


# --- all gammapy sub-packages --- #
import gammapy
print(f"gammapy: v{gammapy.__version__}")
from gammapy.data import DataStore
from gammapy.datasets import Datasets, SpectrumDataset
from gammapy.estimators import FluxPointsEstimator, LightCurveEstimator
from gammapy.makers import ReflectedRegionsBackgroundMaker, SafeMaskMaker, SpectrumDatasetMaker, WobbleRegionsFinder

from gammapy.maps import Map, MapAxis, RegionGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import ExpCutoffPowerLawSpectralModel, LogParabolaSpectralModel, PowerLawSpectralModel
from gammapy.modeling.models import SkyModel, create_crab_spectral_model
from gammapy.visualization import plot_spectrum_datasets_off_regions
from gammapy.estimators import FluxPoints

import naima
from naima.models import ExponentialCutoffBrokenPowerLaw, InverseCompton, Synchrotron
# -------------------------------- #

# creating a folder to save the plots
pltpath = '/fefs/aswg/workspace/juan.jimenez/stereo_analysis/thesis_plots/plots/'
dpi = 200     # resolution of saved images

def compute22():

    # --- other parameters --- #
    # name of the source we are studying
    source_name = 'Crab Nebula'
    # ------------------------ #

    # --- data reduction parameters --- #
    energy_min = '0.040 TeV'
    energy_max = '25. TeV'
    n_bins_pdec = 5

    true_energy_min = '0.01 TeV'
    true_energy_max = '100 TeV'
    n_bins_pdec_true = 10

    n_off_regions = 3
    # --------------------------------- #

    # --- SED function parameters --- #
    sed_type = 'e2dnde'
    yunits   = u.Unit('erg cm-2 s-1')

    crab_model = create_crab_spectral_model('magic_lp')

    # print(crab_model)
    reference_models = {'Crab reference (MAGIC, JHEAp 2015)': crab_model,}
    # ------------------------------- #


    # --- file paths --- #
    # dl3 files
    input_dir = '/fefs/aswg/workspace/juan.jimenez/data/dl3_Crab/'
    fermi_lat_sed_file = '/fefs/aswg/workspace/juan.jimenez/data/other_results/SED_Crab_FermiLAT_Arakawa2020.fits'
    # ------------------ #


    # some colors
    colors = ['darkblue', 'darkorange', 'deeppink', 'darkviolet', 'crimson']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)


    data_stored = DataStore.from_dir(input_dir)

    print(f'\nOpening {len(data_stored.obs_table.to_pandas())} files-runs from {input_dir}.')
    obs_ids = data_stored.obs_table.to_pandas()['OBS_ID'].to_numpy()
    print(f'The runs analysed are:\n{obs_ids}')
    display(data_stored.obs_table.to_pandas().head(4))


    observation = data_stored.get_observations(None, required_irf='point-like')

    # getting the metadata from the first run
    first_observation = observation[0]
    event_meta = first_observation.events.table.meta
    aeff_meta  = first_observation.aeff.meta

    # collecting the target position
    target_position = SkyCoord(
        u.Quantity(event_meta['RA_OBJ'],  u.deg),
        u.Quantity(event_meta['DEC_OBJ'], u.deg),
        frame='icrs',
    )

    # global theta cuts
    if 'RAD_MAX' in aeff_meta:
        # get the global theta cut used for creating the IRFs
        on_region_radius = aeff_meta['RAD_MAX'] * u.deg

        # use the circle sky region to apply the global theta cut
        on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)
        print(f'\nPosition of the ON region (using global theta cuts): \n', on_region)

    # dynamic theta cuts
    else:
        # use the point sky region to apply dynamic theta cuts
        on_region = PointSkyRegion(target_position)
        print(f'\nPosition of the ON region (using dynamic theta cuts): \n', on_region)


    # ------- energy axis ------- #

    # getting the energy axis
    energy_axis = MapAxis.from_energy_bounds(
        energy_min,
        energy_max,
        nbin=n_bins_pdec,
        per_decade=True,
        name='energy',
    )

    # getting true energy axis (MC data)
    energy_axis_true = MapAxis.from_energy_bounds(
        true_energy_min,
        true_energy_max,
        nbin=n_bins_pdec_true,
        per_decade=True,
        name='energy_true',
    )

    fig, ax = plt.subplots(figsize=(12,1))
    ax.plot(energy_axis.edges, np.zeros(len(energy_axis.edges)), '+--', lw=1, ms=10)
    ax.set_xscale('log')
    ax.set_title('Energy axis divisions')
    ax.set_yticks([])
    ax.set_xlabel('E [TeV]')
    plt.show()

    # ------ creating the makers ------ #

    # iterate over datasets
    print(f'\nCreating geometry...')

    # create ON region geometry
    on_geom = RegionGeom.create(region=on_region, axes=[energy_axis])

    dataset_empty = SpectrumDataset.create(geom=on_geom, energy_axis_true=energy_axis_true)

    # create a spectrum dataset maker
    dataset_maker = SpectrumDatasetMaker(
        containment_correction=False,
        selection=['counts', 'exposure', 'edisp'],
        use_region_center=True,
    )

    # create a background maker
    print(f'Number of OFF regions: {n_off_regions}')

    # finding the OFF regions
    region_finder = WobbleRegionsFinder(n_off_regions=n_off_regions)
    bkg_maker     = ReflectedRegionsBackgroundMaker(region_finder=region_finder)

    # create a safe mask maker
    safe_mask_maker = SafeMaskMaker(methods=['aeff-max'], aeff_percent=10)




    datasets = Datasets()                                  
    counts   = Map.create(skydir=target_position, width=3)

    # Loop over every observation
    print('Running the makers...')

    n_observations = len(observation)

    for i_obs, obs in enumerate(observation):

        if (i_obs % 10) == 0:
            print(f'{i_obs}/{n_observations}')

        obs_id = obs.obs_id

        # Fill the number of events in the map
        counts.fill_events(obs.events)

        # Run the makers to the observation data
        dataset        = dataset_maker.run(dataset_empty.copy(name=str(obs_id)), obs)
        dataset_on_off = bkg_maker.run(dataset, obs)
        dataset_on_off = safe_mask_maker.run(dataset_on_off, obs)

        datasets.append(dataset_on_off)

    print(f'{n_observations}/{n_observations}')

    # Get the information table
    info_table = datasets.info_table(cumulative=True)

    # Show the table
    display(info_table.to_pandas().head(5))



    fig = plt.figure(figsize=(8,8))

    # Plot the count map
    ax = counts.plot(add_cbar=True)

    # Plot the ON position
    on_geom.plot_region(ax=ax, edgecolor='r', lw=3)

    # Plot the OFF positions (only the first part of observations)
    plot_spectrum_datasets_off_regions(datasets[:10], ax, edgecolor='w', legend=True, 
                                      prop_cycle=plt.cycler(color=list('w'*100)))
    plot_spectrum_datasets_off_regions(datasets[10:], ax, edgecolor='w', legend=False, 
                                      prop_cycle=plt.cycler(color=list('w'*100)))
    ax.grid()
    plt.tick_params(axis='x', which='both', bottom=True, top=False)
    plt.tick_params(axis='y', which='both', left=True, right=False)
    plt.show()



    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

    # plot the number of events along the livetime
    ax1.plot(info_table['livetime'].to('h'), info_table['excess'],     marker='+', ls='-', lw=2, c='darkblue', ms=7)
    ax1.plot(info_table['livetime'].to('h'), info_table['background'], marker='+', ls='-', lw=2, c='darkorange', ms=7)

    # Plot the significance along the livetime
    ax2.plot(info_table['livetime'].to('h'), info_table['sqrt_ts'], marker='+', ls='-', lw=2, c='darkblue', ms=7)

    # plot the number of events along the livetime
    ax3.plot(info_table['livetime'].to('h'), info_table['excess_rate'],     marker='+', ls='-', lw=2, c='darkblue', ms=7)
    ax3.plot(info_table['livetime'].to('h'), info_table['background_rate'], marker='+', ls='-', lw=2, c='darkorange', ms=7)

    ax1.plot([], [], 's', ms=14, color='darkblue', label='Excess')
    ax1.plot([], [], 's', ms=14, color='darkorange', label='Background')

    fig.suptitle(f'Excess significance plot')
    ax1.set_ylabel('Number of events')
    ax2.set_ylabel('Sqrt(TS)')
    ax3.set_ylabel('Event rate [event/s]')
    ax1.legend()
    for ax in [ax1, ax2, ax3]:    
        ax.set_xlabel('Livetime [h]')
        ax.grid()
    fig.tight_layout()
    plt.show()



    spectral_model = LogParabolaSpectralModel(
        amplitude=u.Quantity(5e-12, unit='cm-2 s-1 TeV-1'),
        alpha=2,
        beta=0.1,
        reference=u.Quantity(1, unit='TeV')
    )

    sky_model = SkyModel(spectral_model=spectral_model.copy(), name=source_name)

    # add the model to the stacked dataset

    stacked_dataset = datasets.stack_reduce()
    stacked_dataset.models = [sky_model]

    # Create a fit object to run on the datasets
    fit = Fit()

    result = fit.run(datasets=stacked_dataset)

    # Keep the best fit model
    best_fit_model = stacked_dataset.models[0].spectral_model.copy()

    # Show the fitted parameters
    display(stacked_dataset.models.to_parameters_table().to_pandas())


    # colors = ['mediumvioletred', 'mediumblue', 'deeppink', 'darkviolet', 'crimson']
    # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
    # plt.rcParams['lines.linewidth'] = 2
    # plt.rcParams['legend.frameon'] = False


    fig, ax = plt.subplots(figsize=(9, 5))

    # plot the number of excess and predicted events
    kwargs_residuals = {'color': 'k'}
    ax_spectrum, ax_residuals = stacked_dataset.plot_fit(kwargs_residuals=kwargs_residuals)

    ax_spectrum.set_ylabel('Counts')
    ax_residuals.set_ylabel('Residuals')
    ax_spectrum.grid(which='both', color='lightgray')

    plt.setp(ax_spectrum.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=0)

    ax_residuals.grid(which='both', color='lightgray')


    # plt.savefig(f'{pltpath}migration-matrix-1.pdf', bbox_inches='tight', dpi=dpi)
    plt.show()

    # create a flux point estimator
    flux_points_estimator = FluxPointsEstimator(energy_edges=energy_axis.edges, source=source_name, selection_optional='all')

    # Run the flux point estimator to the datasets
    print('Running the flux points estimator...')

    flux_points = flux_points_estimator.run(datasets=stacked_dataset)

    # show the flux points table
    display(flux_points.to_table(sed_type='e2dnde', formatted=True)[:5])


    frozen_params = ['alpha', 'beta']

    sky_model = SkyModel(spectral_model=best_fit_model.copy(), name=source_name)

    # freeze the spectral parameters
    for param in frozen_params:
        sky_model.parameters[param].frozen = True

    # Add the model to the datasets
    datasets.models = [sky_model]
    print(sky_model)


    energy_edges = energy_axis.edges[[1,-1]]

    time_intervals = None # `None` automatically makes a 'run-wise' LC.

    # create a light curve estimator
    light_curve_estimator = LightCurveEstimator(energy_edges=energy_edges,
                                                time_intervals=time_intervals,
                                                source=source_name,
                                                selection_optional='all'
    )

    # run the light curve estimator to the datasets

    print(f'\nRunning the light curve estimator...')
    light_curve = light_curve_estimator.run(datasets=datasets)

    # show the light curve table
    display(light_curve.to_table(sed_type='flux', format='lightcurve')[:5])



    fig, ax = plt.subplots(figsize=(12, 5))

    # plot the light curve
    light_curve.plot(ax=ax,
                    sed_type='flux', 
                    label=f'LST-1 + MAGIC (this work)', 
                    color=colors[0]
    )

    mean_flux = np.mean(light_curve.to_table(format='lightcurve', sed_type='flux')['flux'])
    stdv_flux = np.std(light_curve.to_table(format='lightcurve', sed_type='flux')['flux'])/2
    ax.axhline(mean_flux)
    ax.axhspan(mean_flux-stdv_flux, mean_flux+stdv_flux, color='royalblue', alpha=0.4)


    # plot the reference flux
    for label, model in reference_models.items():

        integ_flux = model.integral(energy_edges[0], energy_edges[1])
        ax.axhline(integ_flux, label=label, linestyle='--', lw=2, color='deeppink')

    energy_range = f'{energy_edges[0]:.3f} < $E$ < {energy_edges[1]:.1f}'

    ax.set_title(f'Light curve of {source_name} ({energy_range})')
    ax.set_ylabel('Flux [cm$^{-2}$ s$^{-1}$]')
    ax.set_xlabel('Time [date]')

    # ax.set_xlim(18570, 18590)
    ax.legend()
    ax.set_yscale('linear')
    ax.grid()

    ax.xaxis.set_major_locator(MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%b'))

    plt.show()

    table_LC = light_curve.to_table(sed_type='flux', format='lightcurve')
    flux     = np.array(table_LC['flux'], dtype=float) 
    flux     = np.array([f[0] for f in flux]) * 1e9
    flux_err = np.array(table_LC['flux_err'], dtype=float) 
    flux_err = np.array([f[0] for f in flux_err]) * 1e9
    time     = np.array(table_LC['time_min'], dtype=float)

    reference_date = datetime(1858, 11, 17)
    delta = [timedelta(days=t) for t in time]
    date  = [reference_date + d for d in delta]

    mean_flux = np.mean(flux)
    stdv_flux = np.std(flux)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

    for ax in [ax1, ax2]:
        # plot the light curve
        ax.errorbar(date, flux, yerr=flux_err, marker='o', lw=2, ls='', color=colors[0], label=f'LST-1 + MAGIC (this work)')

        # plot the reference flux
        for label, model in reference_models.items():

            integ_flux, integ_flux_error = np.array(model.integral_error(energy_edges[0], energy_edges[1]).value) * 1e9

            ax.axhline(integ_flux, label=label, linestyle='--', color='deeppink', lw=2, zorder=10)
            ax.axhspan(integ_flux-integ_flux_error, integ_flux+integ_flux_error, color='deeppink', alpha=0.4)

        ax.axhline(mean_flux, lw=2, label='Best constant fit (this work)')
        ax.axhspan(mean_flux-stdv_flux, mean_flux+stdv_flux, color='royalblue', alpha=0.4)
        ax.grid()

        ax.xaxis.set_tick_params(rotation=30)

    energy_range = f'{energy_edges[0].to(u.GeV):.0f} < $E$ < {energy_edges[1]:.0f}'
    print(f'Light curve ({energy_range})')
    fig.suptitle(f'{energy_range}')
    ax1.set_ylabel('Flux [cm$^{-2}$ s$^{-1}$] $\\times \ 10^{-9}$')
    ax2.set_ylabel('')

    ax1.xaxis.set_major_locator(MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%b'))
    ax2.xaxis.set_major_locator(DayLocator(interval=1))
    ax2.xaxis.set_major_formatter(DateFormatter('%Y-%b-%d'))
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    for tick in ax2.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False); tick.tick2line.set_visible(False)
        tick.label1.set_visible(False); tick.label2.set_visible(False)
    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, lw=2)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)

    fig.subplots_adjust(wspace=-1)

    ax1.set_xlim(18575, 18717)
    ax2.set_xlim(19052.7, 19057.3)

    ax2.legend()

    fig.tight_layout()

    plt.savefig(f'{pltpath}Crab-flux.pdf', bbox_inches='tight', dpi=dpi)
    plt.show()

    extra_points_file = '/fefs/aswg/workspace/juan.jimenez/data/other_results/CrabNebula_spectrum.ecsv'
    df_extra = pd.read_csv(extra_points_file, sep=' ', comment='#')

    df_extra['energy'] = df_extra['energy'] * 1e6

    paper_abvr = ['baars', 'macias', 'mezger', 'bandiera', 'kirshner', 'hennessy', 'xmm', 'spi', 'fermi_33months', 'magic', 'hegra', 'hess']

    extra_flux_series = {'baars':'Baars & Hartsuijker (1972)',
                        'macias':'Macías-Pérez et al. (2010)',
                        'mezger':'P.G. Mezger et al. (1986)',
                        'bandiera':'R. Bandiera et al. (2002)',
                        'kirshner':'R.P. Kirshner (1974)',
                        'hennessy':'G.S. Hennessy et al. (1992)',
                        'xmm':'R. Willingale et al. (2000)',
                        'spi':'E. Jourdain & J. P. Roques (2009)',
                        'fermi_33months':'A.A. Abdo et al. (2010)',
                        'magic':'J. Aleksić et al. (2015)',
                        'hegra':'F.A. Aharonian et al. (2000)',
                        'hess':'F.A. Aharonian et al. (2006)'}
    extra_flux_series = {'baars':'Baars et al. (1972) & Macías et al. (2010)',
                        'macias':None,
                        'mezger':'Mezger et al. (1986) & Bandiera et al. (2002)',
                        'bandiera':None,
                        'kirshner':'Kirshner (1974) & Hennessy et al. (1992)',
                        'hennessy':None,
                        'xmm':'R. Willingale et al. (2000)',
                        'spi':'E. Jourdain & J. P. Roques (2009)',
                        'fermi_33months':'Fermi-LAT (A.A. Abdo et al. 2010)',
                        'magic':'MAGIC (J. Aleksić et al. 2015)',
                        'hegra':'HEGRA (F.A. Aharonian et al. 2000)',
                        'hess':'HESS (F.A. Aharonian et al. 2006)'}

    extra_flux_colors = {'baars':'maroon',
                        'macias':'maroon',
                        'mezger':'crimson',
                        'bandiera':'crimson',
                        'kirshner':'#01D776',
                        'hennessy':'#01D776',
                        'xmm':'#00D6C2',
                        'spi':'#00ffff',
                        'fermi_33months':'cornflowerblue',
                        'magic':'c',
                        'hegra':'darkviolet',
                        'hess':'deeppink'}

    e_extra = [df_extra[df_extra['paper'] == p]['energy'].to_numpy(dtype=float) for p in paper_abvr]
    labels_extra = [extra_flux_series[p] for p in paper_abvr]                      
    colors_extra = [extra_flux_colors[p] for p in paper_abvr]
    e2dnde_extra = [df_extra[df_extra['paper'] == p]['flux'].to_numpy(dtype=float) for p in paper_abvr]
    e2dnde_err_extra = [df_extra[df_extra['paper'] == p]['flux_error'].to_numpy(dtype=float) for p in paper_abvr]

    fermi_flux_points = FluxPoints.read(fermi_lat_sed_file)
    table_fer = fermi_flux_points.to_table(sed_type='e2dnde', formatted=True)

    e_fer = u.Quantity(table_fer['e_ref'], 'MeV').to('TeV')

    e_edges_fer = geom.compute_bin_edges(e_fer.value)*u.TeV
    e_err_fer   = (e_edges_fer[:-1] + e_edges_fer[1:]) / 2

    e2dnde_fer         = u.Quantity(table_fer['e2dnde'],     'MeV / (cm2 s)').to('erg / (cm2 s)')
    e2dnde_err_fer     = u.Quantity(table_fer['e2dnde_err'], 'MeV / (cm2 s)').to('erg / (cm2 s)')


    table_lp1 = flux_points.to_table(sed_type='e2dnde', formatted=True)

    e_d1 = u.Quantity(table_lp1['e_ref'], 'TeV')

    e_left_d1 = u.Quantity(table_lp1['e_min'], 'TeV')
    e_right_d1 = u.Quantity(table_lp1['e_max'], 'TeV')
    e_err_d1 = [e_d1 - e_left_d1, e_right_d1 - e_d1]

    e2dnde_d1     = u.Quantity(table_lp1['e2dnde'],     'TeV / (cm2 s)').to('erg / (cm2 s)')
    e2dnde_err_d1 = u.Quantity(table_lp1['e2dnde_err'], 'TeV / (cm2 s)').to('erg / (cm2 s)')

    fit_params_lp1 = best_fit_model.parameters.value
    fit_params_lp1 = [u.Quantity(fit_params_lp1[0], '1 / (cm2 s TeV)'), 1*u.TeV, fit_params_lp1[2], fit_params_lp1[3]]

    e2dnde_lp1 =  e_d1 * e_d1 * model.evaluate(e_d1, *fit_params_lp1)

    residuals_lp1     = (e2dnde_lp1.to('TeV / (cm2 s)') - e2dnde_d1.to('TeV / (cm2 s)')) / e2dnde_d1.to('TeV / (cm2 s)') * 100
    residuals_err_lp1 = e2dnde_err_d1 / e2dnde_d1 * 100

    print(f'Log Par fit-1 parameters: {fit_params_lp1}')


    #############################################
    energy_bounds = energy_axis.edges[[0, -1]]

    #############################################

    fig, (ax1, axb1) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3.5, 1]})


    # --- flux points --- #
    flux_points.plot(ax=ax1,
                     sed_type=sed_type, 
                     label=f'LST-1 + MAGIC (this work)',
                     color=colors[0],
                     zorder=5
    )


    # --- best fit model --- #
    best_fit_model.plot(ax=ax1,
                        energy_bounds=energy_bounds,
                        sed_type=sed_type,
                        yunits=yunits,
                        color=colors[0],
                        label='Best fit (this work)',
                        zorder=5
    )

    # --- best fit model error --- #
    best_fit_model.plot_error(ax=ax1,
                              energy_bounds=energy_bounds,
                              sed_type=sed_type,
                              yunits=yunits,
                              facecolor=colors[0],
                              zorder=4
    )


    # --- reference spectra --- #
    for label, model in reference_models.items():
        model.plot(ax=ax1,
                   energy_bounds=energy_bounds,
                   sed_type=sed_type,
                   yunits=yunits,
                   label=label,
                   linestyle='--',
                   color='deeppink'
                  )

    # --- other data points to plot --- #
    for i in range(len(e_extra)):
        if paper_abvr[i] in ['hess', 'hegra', 'magic']:
            ax1.errorbar(e_extra[i]*1e-12, e2dnde_extra[i], yerr=e2dnde_err_extra[i], label=labels_extra[i],
                        marker='.', ls='', color=colors_extra[i], zorder=3, ms=3)


    # --- error --- #
    axb1.errorbar(e_d1[:-1], residuals_lp1[:-1], yerr=residuals_err_lp1[:-1], xerr=[e_err_d1[0][:-1], e_err_d1[1][:-1]], 
                 marker='.', ms=8, ls='', color='k') 
    axb1.axhline(0, color='k', ls='-', lw=1)


    # twin axes ############
    axt1 = ax1.twiny() 

    # plot limits ##########
    xlimsE = np.array([0.03, 30])
    xlimsV = xlimsE * 2.418e14 * 1e12
    for axe in [ax1, axb1]:
        axe.set_xlim(*xlimsE)
    ax1.loglog()
    axb1.set_xscale('log')
    axt1.set_xlim(*xlimsV)

    ax1.set_ylim(1.4e-12, 1.3e-10)
    axt1.set_xscale('log')

    # labels ###############
    ax1.set_ylabel('$E^2\\frac{dN}{dE}$ [erg cm$^{-2}$ s$^{-1}$]')
    axt1.set_xlabel('Frequency [Hz]', labelpad=10)
    axb1.set_ylabel('Residuals %')
    axb1.set_xlabel('E [TeV]')

    # legends ##############
    for axe in [ax1, axb1]:
        axe.grid(which='both')
    ax1.legend(loc='lower left', fontsize=12, )

    # ticks ################
    ax1.set_xticklabels([])


    fig.subplots_adjust(wspace=0, hspace=0)


    plt.savefig(f'{pltpath}logpar-thiswork.pdf', bbox_inches='tight', dpi=dpi)
    plt.show()


    # joint points
    e_d2 = np.concatenate([e_fer[:-1], e_d1[2:]])[6:-4]

    # temporal assignment
    e_err_right_f = e_err_fer                                 
    e_err_left_f  = geom.compute_left_errors(e_fer.value, e_err_fer.value)*u.TeV 
    e_err_left_f, e_err_right_f = geom.compute_left_right_errors(e_fer.value)
    e_err_left_f, e_err_right_f = e_err_left_f*u.TeV, e_err_right_f*u.TeV


    e_err_right_d2 = np.concatenate([e_err_right_f[:-1], e_err_d1[1][2:]])[6:-4]
    e_err_left_d2  = np.concatenate([e_err_left_f[:-1],  e_err_d1[0][2:]])[6:-4]
    e_err_d2       = [e_err_left_d2, e_err_right_d2]

    e2dnde_d2     = u.Quantity(np.concatenate([e2dnde_fer[:-1], e2dnde_d1[2:]])[6:-4], 'erg / (cm2 s)').to('TeV / (cm2 s)')
    e2dnde_err_d2 = u.Quantity(np.concatenate([e2dnde_err_fer[:-1], e2dnde_err_d1[2:]])[6:-4], 'erg / (cm2 s)').to('TeV / (cm2 s)')

    # finding the best fit parameters
    fit_params_lp2, _ = curve_fit(geom.logpar, e_d2.value, e2dnde_d2.value, p0=best_fit_model.parameters.value)
    fit_params_lp2    = [u.Quantity(fit_params_lp2[0], '1 / (cm2 s TeV)'), 1*u.TeV, fit_params_lp2[2], fit_params_lp2[3]]

    e2dnde_lp2 =  e_d2 * e_d2 * model.evaluate(e_d2, *fit_params_lp2)

    residuals_lp2     = (e2dnde_lp2.to('TeV / (cm2 s)') - e2dnde_d2.to('TeV / (cm2 s)')) / e2dnde_d2.to('TeV / (cm2 s)') * 100
    residuals_err_lp2 = e2dnde_err_d2 / e2dnde_d2 * 100

    print(f'Log Par fit-2 parameters: {fit_params_lp2}')


    #############################################
    energy_bounds = energy_axis.edges[[0, -1]]

    e_lspace = np.logspace(-2.6, 1.4, 100) * u.TeV
    #############################################

    fig, (ax1, axb1) = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [3.5, 1]})

    # --- flux points --- #
    flux_points.plot(ax=ax1,
                        sed_type=sed_type, 
                        label=f'LST-1 + MAGIC (this work)',
                        color=colors[0],
    )

    # --- best fit model --- #
    best_fit_model.plot(ax=ax1,
                        energy_bounds=energy_bounds,
                        sed_type=sed_type,
                        yunits=yunits,
                        color=colors[0],
                        label='Best fit (this work)'
    )


    # --- fermi reference --- #
    fermi_flux_points.plot(
                        label='Fermi-LAT (Arakawa et al. 2020)',
                        color='royalblue',
                        marker='^',
                        zorder=3,
                        sed_type='e2dnde',
                        ax=ax1,
                        markersize=4
    )

    # --- plot the new fit with joint data --- #
    ax1.plot(e_lspace.value, (e_lspace * e_lspace * best_fit_model.evaluate(e_lspace, *fit_params_lp2)).to('erg / (cm2 s)'), 
            color='crimson',
            label='Best fit (Fermi-LAT + this work)')

    # --- plot extra experimental points --- #
    for i in range(len(e_extra)):
        if paper_abvr[i] in ['fermi_33months', 'hess', 'hegra', 'magic']:
            ax1.errorbar(e_extra[i]*1e-12, e2dnde_extra[i], yerr=e2dnde_err_extra[i], label=labels_extra[i],
                        marker='.', ls='', color=colors_extra[i], zorder=3, ms=3)

    # --- error --- #
    axb1.errorbar(e_d2, residuals_lp2, yerr=residuals_err_lp2, xerr=e_err_d2, marker='.', ms=8, ls='', color='k', label='Fermi-LAT + this work', zorder=4) 
    axb1.axhline(0, color='k', ls='-', lw=1, zorder=1)
    axb1.errorbar(e_d1[:-1], residuals_lp1[:-1], yerr=residuals_err_lp1[:-1], xerr=[e_err_d1[0][:-1], e_err_d1[1][:-1]], 
                 marker='.', ms=8, ls='', color='gray', zorder=3, label='Only this work')


    # twin axes ############
    axt1 = ax1.twiny() 

    # plot limits ##########
    xlimsE = np.array([3e-5, 1e2])
    xlimsV = xlimsE * 2.418e14 * 1e12
    for axe in [ax1, axb1]:
        axe.set_xlim(*xlimsE)
    ax1.loglog()
    axb1.set_xscale('log')
    axt1.set_xlim(*xlimsV)

    ax1.set_ylim(1e-12, 3e-10)
    axt1.set_xscale('log')

    # labels ###############
    ax1.set_ylabel('$E^2\\frac{dN}{dE}$ [erg cm$^{-2}$ s$^{-1}$]')
    axt1.set_xlabel('Frequency [Hz]', labelpad=10)
    axb1.set_ylabel('Residuals %', labelpad=16)
    axb1.set_xlabel('E [TeV]')

    # legends ##############
    for axe in [ax1, axb1]:
        axe.grid(which='both', color='0.8', zorder=-100)
        axe.legend(loc='lower left', fontsize=12, )

    # ticks ################
    ax1.set_xticklabels([])
    [line.set_zorder(100) for line in ax1.lines]
    [line.set_zorder(100) for line in axb1.lines]

    fig.subplots_adjust(wspace=0, hspace=0)


    plt.savefig(f'{pltpath}logpar-fermi.pdf', bbox_inches='tight', dpi=dpi)
    plt.show()


    def ff1(e):
        return e * e * best_fit_model.evaluate(e*u.TeV, *fit_params_lp1)

    def ff2(e):
        return e * e * best_fit_model.evaluate(e*u.TeV, *fit_params_lp2)

    max1 = optimize.fmin(lambda x: -ff1(x), 1)[0]
    max2 = optimize.fmin(lambda x: -ff2(x), 1)[0]

    print(f'\n\nPeak of parabola at:\nMAGIC+LST-1\t\t-->\t{max1:.3f} TeV\nFermi+MAGIC+LST-1\t-->\t{max2:.3f} TeV')


    def SYN_IC_LOG(energy, ampl=1, a1=1, a2=1, b=1, ecut=1, ebr=1, Bmag=1):
        energy=10**energy
        energy = u.Quantity(energy, 'eV')
        ECBPL = ExponentialCutoffBrokenPowerLaw(amplitude= ampl * 3.699e36 / u.eV, e_0=1 * u.TeV, e_break=ebr * 0.265 * u.TeV, 
                                                alpha_1=a1*1.5, alpha_2=a2*3.233, e_cutoff=ecut * 1863 * u.TeV, beta=b * 2.0,)
        eopts = {'Eemax': 50 * u.PeV, 'Eemin': 0.1 * u.GeV}
        SYN = Synchrotron(ECBPL, B=Bmag*125 * u.uG, Eemax=50 * u.PeV, Eemin=0.1 * u.GeV)
        Rpwn = 2.1 * u.pc
        Esy = np.logspace(-7, 9, 100) * u.eV
        Lsy = SYN.flux(Esy, distance=0 * u.cm)  # use distance 0 to get luminosity
        phn_sy = Lsy / (4 * np.pi * Rpwn ** 2 * c) * 2.24
        IC = InverseCompton(ECBPL, seed_photon_fields=['CMB', ['FIR', 70 * u.K, 0.5 * u.eV / u.cm ** 3], ['NIR', 5000 * u.K, 1 * u.eV / u.cm ** 3], ['SSC', Esy, phn_sy],],
                            Eemax=50 * u.PeV, Eemin=0.1 * u.GeV,)    
        return np.log10((IC.sed(energy, 2 * u.kpc) + SYN.sed(energy, 2 * u.kpc)).value)

    def SYN_IC(energy, ampl=1, a1=1, a2=1, b=1, ecut=1, ebr=1, Bmag=1):

        ECBPL = ExponentialCutoffBrokenPowerLaw(amplitude= ampl * 3.699e36 / u.eV, e_0=1 * u.TeV, e_break=ebr * 0.265 * u.TeV, 
                                                alpha_1=a1*1.5, alpha_2=a2*3.233, e_cutoff=ecut * 1863 * u.TeV, beta=b * 2.0,)

        eopts = {'Eemax': 50 * u.PeV, 'Eemin': 0.1 * u.GeV}

        SYN = Synchrotron(ECBPL, B=Bmag*125 * u.uG, Eemax=50 * u.PeV, Eemin=0.1 * u.GeV)

        Rpwn = 2.1 * u.pc
        Esy = np.logspace(-7, 9, 100) * u.eV
        Lsy = SYN.flux(Esy, distance=0 * u.cm)  # use distance 0 to get luminosity
        phn_sy = Lsy / (4 * np.pi * Rpwn ** 2 * c) * 2.24

        IC = InverseCompton(ECBPL, seed_photon_fields=['CMB', ['FIR', 70 * u.K, 0.5 * u.eV / u.cm ** 3], ['NIR', 5000 * u.K, 1 * u.eV / u.cm ** 3], ['SSC', Esy, phn_sy],],
                            Eemax=50 * u.PeV, Eemin=0.1 * u.GeV,)    

        return IC.sed(energy, 2 * u.kpc) + SYN.sed(energy, 2 * u.kpc)


    def ECBPL_electrons(energy, ampl=1, a1=1, a2=1, b=1, ecut=1, ebr=1):

        return ExponentialCutoffBrokenPowerLaw.eval(e=energy, amplitude= ampl * 3.699e36 / u.eV, e_0=1 * u.TeV, e_break=ebr * 0.265 * u.TeV, 
                          alpha_1=a1*1.5, alpha_2=a2*3.233, e_cutoff=ecut * 1863 * u.TeV, beta=b * 2.0,)

    def IC_only(ampl=1, a1=1, a2=1, b=1, ecut=1, ebr=1, Bmag=1):

        ECBPL = ExponentialCutoffBrokenPowerLaw(
            amplitude=ampl * 3.699e36 / u.eV, e_0=1 * u.TeV, e_break=ebr * 0.265 * u.TeV, alpha_1=a1 * 1.5, alpha_2=a2 * 3.233, e_cutoff=ecut * 1863 * u.TeV, beta=b * 2.0)
        eopts = {'Eemax': 50 * u.PeV, 'Eemin': 0.1 * u.GeV}

        SYN = Synchrotron(ECBPL, B=Bmag * 125 * u.uG, Eemax=50 * u.PeV, Eemin=0.1 * u.GeV)

        Rpwn = 2.1 * u.pc
        Esy = np.logspace(-7, 9, 100) * u.eV
        Lsy = SYN.flux(Esy, distance=0 * u.cm)  # use distance 0 to get luminosity
        phn_sy = Lsy / (4 * np.pi * Rpwn ** 2 * c) * 2.24

        IC = InverseCompton(
            ECBPL,
            seed_photon_fields=[
                'CMB',
                ['FIR', 70 * u.K, 0.5 * u.eV / u.cm ** 3],
                ['NIR', 5000 * u.K, 1 * u.eV / u.cm ** 3],
                ['SSC', Esy, phn_sy],
            ],
            Eemax=50 * u.PeV,
            Eemin=0.1 * u.GeV,)
        return IC


    perform_fit = False

    aux.createdir('data')

    e_d3 = np.concatenate([np.concatenate(e_extra)[:-30], e_d1.to('eV')[2:-1].value])*u.eV

    e2dnde_d3 = u.Quantity(np.concatenate([np.concatenate(e2dnde_extra)[:-30], e2dnde_d1.to('erg / (cm2 s)')[2:-1].value]), 'erg / (cm2 s)')
    e2dnde_err_d3 = u.Quantity(np.concatenate([np.concatenate(e2dnde_err_extra)[:-30], e2dnde_err_d1.to('erg / (cm2 s)')[2:-1].value]), 'erg / (cm2 s)')


    if perform_fit == True:
        fit_params_d3, _ = curve_fit(SYN_IC_LOG, np.log10(e_d3.value), np.log10(e2dnde_d3.value), sigma=np.log10(e2dnde_err_d3.value), p0=np.repeat(1, 7), bounds=[np.repeat(0.5, 7), np.repeat(2, 7)])
        np.savetxt(f'data/fit_params_d3_{source_name}.txt', fit_params_d3)
    fit_params_d3 = np.loadtxt(f'data/fit_params_d3_{source_name}.txt')


    for fp, name in zip([np.repeat(1, 7), fit_params_d3], ['Naima', 'This Work']):
        print(f'\nOPTIMAL PARAMETERS for {name}:\nFor the electron distribution:\nAmplitude\t\t= {fp[0]* 3.699e36 / u.eV:.2e}\nalpha before break\t= {fp[1]*1.5:.2f}')
        print(f'alpha after break\t= {fp[2]*3.233:.2f}\nbeta\t\t\t= {fp[3]* 2.0:.2f}\ncutoff energy\t\t= {fp[4] * 1863 * u.TeV:.2f}')
        print(f'break energy\t\t= {fp[5]* 0.265 * u.TeV:.2f}\nOthers:\nMagnetic Field\t\t={fp[6]*125 * u.uG:.2f}')

    e2dnde_syncic_1 =  SYN_IC(e_d3, *fit_params_d3)
    e2dnde_syncic_naima =  SYN_IC(e_d3, *np.repeat(1, 7))

    residuals_syncic_1     = (e2dnde_syncic_1.to('eV / (cm2 s)') - e2dnde_d3.to('eV / (cm2 s)')) / e2dnde_d3.to('eV / (cm2 s)') * 100
    residuals_err_syncic_1 = e2dnde_err_d3 / e2dnde_d3 * 100

    residuals_syncic_naima     = (e2dnde_syncic_naima.to('eV / (cm2 s)') - e2dnde_d3.to('eV / (cm2 s)')) / e2dnde_d3.to('eV / (cm2 s)') * 100
    residuals_err_syncic_naima = e2dnde_err_d3 / e2dnde_d3 * 100


    table = flux_points.to_table(sed_type='e2dnde', formatted=True)

    # change to eV
    for lab in ['e_ref', 'e_min', 'e_max']:
        table[lab] = table[lab].to('eV')
    table[:5]

    flux_points_eV = FluxPoints.from_table(table, sed_type='e2dnde')

    ##############################################

    energy = np.logspace(-7, 15, 200) * u.eV
    energy_bounds = energy_axis.edges[[0, -1]]

    ##############################################


    fig, (ax1, axb1) = plt.subplots(2, 1, figsize=(15, 8.5), gridspec_kw={'height_ratios': [3.5, 1]})

    # --- flux points --- #
    flux_points_eV.plot(ax=ax1,
                      sed_type=sed_type, 
                      label=f'LST-1+MAGIC (this work)',
                      color=colors[0],
                      zorder=100,
                      marker='.'
    )

    # --- fermi reference --- #
    fermi_flux_points.plot(
        label='Fermi-LAT (Arakawa et al. 2020)',
        color='royalblue',
        marker='^',
        zorder=-3,
        sed_type='e2dnde',
        ax=ax1,
    )

    # --- computed model emission --- #
    ax1.plot(energy, SYN_IC(energy, *fit_params_d3),
        lw=1.5, c='k', zorder=-10,
    )
    ax1.plot(energy, SYN_IC(energy, *np.repeat(1, 7)),
        lw=1.5, c='k', ls='--', zorder=-10,
    )

    # --- components of the SI --- #
    for i, seed, ls in zip(range(4), ['CMB', 'FIR', 'NIR', 'SSC'], ['--', '-.', ':', '-']):
        ax1.loglog(energy, IC_only(*fit_params_d3).sed(energy, 2 * u.kpc, seed=seed),
            lw=1.5, c='gray', ls=ls, zorder=-1, alpha=0.5,
        )

    # --- plotting extra data --- #
    for i in range(len(e_extra)):
        if paper_abvr[i] not in ['hess', 'magic']:
            ax1.errorbar(e_extra[i], e2dnde_extra[i], yerr=e2dnde_err_extra[i], xerr=0, label=labels_extra[i],
                        marker='o', ls='', color=colors_extra[i], zorder=10, ms=2)


    # --- labeling IC components --- #
    ax1.text(4.5e7, 6e-12, 'CMB', color='gray', fontsize=11, rotation=55, ha='center', va='center')
    ax1.text(4e8, 1e-11, 'SSC', color='gray', fontsize=11, rotation=52, ha='center', va='center')
    ax1.text(3e8, 2e-12, 'FIR', color='gray', fontsize=11, rotation=65, ha='center', va='center')
    ax1.text(3e10, 6e-12, 'NIR', color='gray', fontsize=11, rotation=48, ha='center', va='center')

    # --- error --- #
    axb1.errorbar(e_d3, residuals_syncic_1, yerr=residuals_err_syncic_1, marker='.', ms=4, ls='', color='k', lw=1, label='This work', zorder=10) 
    axb1.errorbar(e_d3, residuals_syncic_naima, yerr=residuals_err_syncic_naima, marker='.', ms=4, ls='', color='gray', lw=1,label='Naima standard', zorder=6) 
    axb1.axhline(0, color='k', ls='-', lw=1, zorder=0)


    # twin axes ############
    axt1 = ax1.twiny() 

    # plot limits ##########
    xlimsE = np.array([5e-8, 1e15])
    xlimsV = xlimsE * 2.418e14 * 1e12
    for axe in [ax1, axb1]:
        axe.set_xlim(*xlimsE)
    ax1.loglog()
    axb1.set_xscale('log')
    axt1.set_xlim(*xlimsV)

    ax1.set_ylim(8e-13, 1e-7)
    axt1.set_xscale('log')

    # labels ###############
    ax1.set_ylabel('$E^2\\frac{dN}{dE}$ [erg cm$^{-2}$ s$^{-1}$]')
    axt1.set_xlabel('Frequency [Hz]', labelpad=12)
    axb1.set_ylabel('Residuals %')
    axb1.set_xlabel('E [eV]')

    # legends ##############
    for axe in [ax1, axb1]:
        axe.grid()
        [line.set_zorder(-10) for line in axe.lines]
    ax1.legend(loc=(0.23, 0.04), fontsize=12,)
    axb1.legend(loc=2, frameon=False)
    axt1.plot([], [], lw=2, color='k', ls='--', label='Naima standard', zorder=0)
    axt1.plot([], [], lw=2, color='k', ls='-', label='This work', zorder=0)
    axt1.legend(loc=1, frameon=False, fontsize=12)

    # ticks ################
    ax1.set_xticklabels([])
    axb1.set_yticks([-100, 0, 100])
    fig.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(f'{pltpath}multiwavelength-fit.pdf', bbox_inches='tight', dpi=dpi)
    plt.show()

    table_vhe  = table[2:].copy()
    df_tm = df_extra.query(f'paper != "ma"')#paper == "spi" or paper == "xmm"

    table_vhe['energy'] = table_vhe['e_ref']
    table_vhe['flux'] = table_vhe['e2dnde']
    table_vhe['flux_error'] = table_vhe['e2dnde_err']

    table_xray = table_vhe.copy()

    for i in range(len(df_tm)-len(table_vhe)):
        table_xray.add_row(np.repeat(0, 21))

    ee = df_tm['energy']
    ff = df_tm['flux']
    fe = df_tm['flux_error']

    table_xray['energy'] = ee*u.TeV
    table_xray['flux']   = ff
    table_xray['flux_error'] = fe

    table_xray['energy'].unit = u.eV
    table_xray['flux'].unit = u.erg / u.s / u.cm / u.cm
    table_xray['flux_error'].unit = u.erg / u.s / u.cm / u.cm

    def trapz_loglog(y, x, axis=-1, intervals=False):
        try:
            y_unit = y.unit
            y = y.value
        except AttributeError:
            y_unit = 1.0
        try:
            x_unit = x.unit
            x = x.value
        except AttributeError:
            x_unit = 1.0

        y = np.asanyarray(y)
        x = np.asanyarray(x)

        slice1 = [slice(None)] * y.ndim
        slice2 = [slice(None)] * y.ndim
        slice1[axis] = slice(None, -1)
        slice2[axis] = slice(1, None)

        slice1 = tuple(slice1)
        slice2 = tuple(slice2)

        if x.ndim == 1:
            shape = [1] * y.ndim
            shape[axis] = x.shape[0]
            x = x.reshape(shape)

        # Compute the power law indices in each integration bin
        b = np.log10(y[slice2] / y[slice1]) / np.log10(x[slice2] / x[slice1])

        # if local powerlaw index is -1, use \int 1/x = log(x); otherwise use
        # normal powerlaw integration
        trapzs = np.where(
            np.abs(b + 1.0) > 1e-10,
            (
                y[slice1]
                * (x[slice2] * (x[slice2] / x[slice1]) ** b - x[slice1])
            )
            / (b + 1),
            x[slice1] * y[slice1] * np.log(x[slice2] / x[slice1]),
        )

        tozero = (y[slice1] == 0.0) + (y[slice2] == 0.0) + (x[slice1] == x[slice2])
        trapzs[tozero] = 0.0

        if intervals:
            return trapzs * x_unit * y_unit

        ret = np.add.reduce(trapzs, axis) * x_unit * y_unit

        return ret

    yval1 = (SYN_IC(energy, *np.repeat(1, 7)))
    res1 = (trapz_loglog(yval1.value/energy.value, energy.value) * yval1.unit).to('eV / (cm2 s)')
    yval2 = (SYN_IC(energy, *fit_params_d3))
    res2 = (trapz_loglog(yval2.value/energy.value, energy.value) * yval2.unit).to('eV / (cm2 s)')

    print(f'Total energy budget:\nNaima model ---> {res1:.3e}\nThis work   ---> {res2:.3e}')




    def SYN(energy, ampl=1, a1=1, a2=1, b=1, ecut=1, ebr=1, Bmag=1):

        ECBPL = ExponentialCutoffBrokenPowerLaw(amplitude= ampl * 3.699e36 / u.eV, e_0=1 * u.TeV, e_break=ebr * 0.265 * u.TeV, 
                                                alpha_1=a1*1.5, alpha_2=a2*3.233, e_cutoff=ecut * 1863 * u.TeV, beta=b * 2.0,)

        eopts = {'Eemax': 50 * u.PeV, 'Eemin': 0.1 * u.GeV}

        SYN = Synchrotron(ECBPL, B=Bmag*125 * u.uG, Eemax=50 * u.PeV, Eemin=0.1 * u.GeV)

        Rpwn = 2.1 * u.pc
        Esy = np.logspace(-7, 9, 100) * u.eV
        Lsy = SYN.flux(Esy, distance=0 * u.cm)  # use distance 0 to get luminosity
        phn_sy = Lsy / (4 * np.pi * Rpwn ** 2 * c) * 2.24

        IC = InverseCompton(ECBPL, seed_photon_fields=['CMB', ['FIR', 70 * u.K, 0.5 * u.eV / u.cm ** 3], ['NIR', 5000 * u.K, 1 * u.eV / u.cm ** 3], ['SSC', Esy, phn_sy],],
                            Eemax=50 * u.PeV, Eemin=0.1 * u.GeV,)    

        return SYN.sed(energy, 2 * u.kpc)

    def IC(energy, ampl=1, a1=1, a2=1, b=1, ecut=1, ebr=1, Bmag=1):

        ECBPL = ExponentialCutoffBrokenPowerLaw(amplitude= ampl * 3.699e36 / u.eV, e_0=1 * u.TeV, e_break=ebr * 0.265 * u.TeV, 
                                                alpha_1=a1*1.5, alpha_2=a2*3.233, e_cutoff=ecut * 1863 * u.TeV, beta=b * 2.0,)

        eopts = {'Eemax': 50 * u.PeV, 'Eemin': 0.1 * u.GeV}

        SYN = Synchrotron(ECBPL, B=Bmag*125 * u.uG, Eemax=50 * u.PeV, Eemin=0.1 * u.GeV)

        Rpwn = 2.1 * u.pc
        Esy = np.logspace(-7, 9, 100) * u.eV
        Lsy = SYN.flux(Esy, distance=0 * u.cm)  # use distance 0 to get luminosity
        phn_sy = Lsy / (4 * np.pi * Rpwn ** 2 * c) * 2.24

        IC = InverseCompton(ECBPL, seed_photon_fields=['CMB', ['FIR', 70 * u.K, 0.5 * u.eV / u.cm ** 3], ['NIR', 5000 * u.K, 1 * u.eV / u.cm ** 3], ['SSC', Esy, phn_sy],],
                            Eemax=50 * u.PeV, Eemin=0.1 * u.GeV,)    

        return IC.sed(energy, 2 * u.kpc)

    def IC_SSC(energy, ampl=1, a1=1, a2=1, b=1, ecut=1, ebr=1, Bmag=1):

        ECBPL = ExponentialCutoffBrokenPowerLaw(amplitude= ampl * 3.699e36 / u.eV, e_0=1 * u.TeV, e_break=ebr * 0.265 * u.TeV, 
                                                alpha_1=a1*1.5, alpha_2=a2*3.233, e_cutoff=ecut * 1863 * u.TeV, beta=b * 2.0,)

        eopts = {'Eemax': 50 * u.PeV, 'Eemin': 0.1 * u.GeV}

        SYN = Synchrotron(ECBPL, B=Bmag*125 * u.uG, Eemax=50 * u.PeV, Eemin=0.1 * u.GeV)

        Rpwn = 2.1 * u.pc
        Esy = np.logspace(-7, 9, 100) * u.eV
        Lsy = SYN.flux(Esy, distance=0 * u.cm)  # use distance 0 to get luminosity
        phn_sy = Lsy / (4 * np.pi * Rpwn ** 2 * c) * 2.24

        IC = InverseCompton(ECBPL, seed_photon_fields=[['SSC', Esy, phn_sy],],
                            Eemax=50 * u.PeV, Eemin=0.1 * u.GeV,)    

        return IC.sed(energy, 2 * u.kpc)

    def IC_CMB(energy, ampl=1, a1=1, a2=1, b=1, ecut=1, ebr=1, Bmag=1):

        ECBPL = ExponentialCutoffBrokenPowerLaw(amplitude= ampl * 3.699e36 / u.eV, e_0=1 * u.TeV, e_break=ebr * 0.265 * u.TeV, 
                                                alpha_1=a1*1.5, alpha_2=a2*3.233, e_cutoff=ecut * 1863 * u.TeV, beta=b * 2.0,)

        eopts = {'Eemax': 50 * u.PeV, 'Eemin': 0.1 * u.GeV}

        SYN = Synchrotron(ECBPL, B=Bmag*125 * u.uG, Eemax=50 * u.PeV, Eemin=0.1 * u.GeV)

        Rpwn = 2.1 * u.pc
        Esy = np.logspace(-7, 9, 100) * u.eV
        Lsy = SYN.flux(Esy, distance=0 * u.cm)  # use distance 0 to get luminosity
        phn_sy = Lsy / (4 * np.pi * Rpwn ** 2 * c) * 2.24

        IC = InverseCompton(ECBPL, seed_photon_fields=['CMB'],
                            Eemax=50 * u.PeV, Eemin=0.1 * u.GeV,)    

        return IC.sed(energy, 2 * u.kpc)

    def IC_FIR(energy, ampl=1, a1=1, a2=1, b=1, ecut=1, ebr=1, Bmag=1):

        ECBPL = ExponentialCutoffBrokenPowerLaw(amplitude= ampl * 3.699e36 / u.eV, e_0=1 * u.TeV, e_break=ebr * 0.265 * u.TeV, 
                                                alpha_1=a1*1.5, alpha_2=a2*3.233, e_cutoff=ecut * 1863 * u.TeV, beta=b * 2.0,)

        eopts = {'Eemax': 50 * u.PeV, 'Eemin': 0.1 * u.GeV}

        SYN = Synchrotron(ECBPL, B=Bmag*125 * u.uG, Eemax=50 * u.PeV, Eemin=0.1 * u.GeV)

        Rpwn = 2.1 * u.pc
        Esy = np.logspace(-7, 9, 100) * u.eV
        Lsy = SYN.flux(Esy, distance=0 * u.cm)  # use distance 0 to get luminosity
        phn_sy = Lsy / (4 * np.pi * Rpwn ** 2 * c) * 2.24

        IC = InverseCompton(ECBPL, seed_photon_fields=[['FIR', 70 * u.K, 0.5 * u.eV / u.cm ** 3],],
                            Eemax=50 * u.PeV, Eemin=0.1 * u.GeV,)    

        return IC.sed(energy, 2 * u.kpc)

    def IC_NIR(energy, ampl=1, a1=1, a2=1, b=1, ecut=1, ebr=1, Bmag=1):

        ECBPL = ExponentialCutoffBrokenPowerLaw(amplitude= ampl * 3.699e36 / u.eV, e_0=1 * u.TeV, e_break=ebr * 0.265 * u.TeV, 
                                                alpha_1=a1*1.5, alpha_2=a2*3.233, e_cutoff=ecut * 1863 * u.TeV, beta=b * 2.0,)

        eopts = {'Eemax': 50 * u.PeV, 'Eemin': 0.1 * u.GeV}

        SYN = Synchrotron(ECBPL, B=Bmag*125 * u.uG, Eemax=50 * u.PeV, Eemin=0.1 * u.GeV)

        Rpwn = 2.1 * u.pc
        Esy = np.logspace(-7, 9, 100) * u.eV
        Lsy = SYN.flux(Esy, distance=0 * u.cm)  # use distance 0 to get luminosity
        phn_sy = Lsy / (4 * np.pi * Rpwn ** 2 * c) * 2.24

        IC = InverseCompton(ECBPL, seed_photon_fields=[['NIR', 5000 * u.K, 1 * u.eV / u.cm ** 3],],
                            Eemax=50 * u.PeV, Eemin=0.1 * u.GeV,)    

        return IC.sed(energy, 2 * u.kpc)


    yval1 = (SYN(energy, *np.repeat(1, 7)))
    res1 = (trapz_loglog(yval1.value/energy.value, energy.value)* yval1.unit).to('eV / (cm2 s)')
    yval2 = (SYN(energy, *fit_params_d3))
    res2 = (trapz_loglog(yval2.value/energy.value, energy.value) * yval2.unit).to('eV / (cm2 s)')

    print(f'Synchrotron energy budget:\nNaima model ---> {res1:.3e}\nThis work   ---> {res2:.3e}')

    yval1 = (IC(energy, *np.repeat(1, 7)))
    res1 = (trapz_loglog(yval1.value/energy.value, energy.value) * yval1.unit).to('eV / (cm2 s)')
    yval2 = (IC(energy, *fit_params_d3))
    res2 = (trapz_loglog(yval2.value/energy.value, energy.value) * yval2.unit).to('eV / (cm2 s)')

    print(f'IC energy budget:\nNaima model ---> {res1:.3e}\nThis work   ---> {res2:.3e}')

    yval1 = (IC_SSC(energy, *np.repeat(1, 7)))
    res1 = (trapz_loglog(yval1.value/energy.value, energy.value) * yval1.unit).to('eV / (cm2 s)')
    yval2 = (IC_SSC(energy, *fit_params_d3))
    res2 = (trapz_loglog(yval2.value/energy.value, energy.value) * yval2.unit).to('eV / (cm2 s)')

    print(f'\n\nIC (SSC) energy budget:\nNaima model ---> {res1:.3e}\nThis work   ---> {res2:.3e}')

    yval1 = (IC_CMB(energy, *np.repeat(1, 7)))
    res1 = (trapz_loglog(yval1.value/energy.value, energy.value) * yval1.unit).to('eV / (cm2 s)')
    yval2 = (IC_CMB(energy, *fit_params_d3))
    res2 = (trapz_loglog(yval2.value/energy.value, energy.value) * yval2.unit).to('eV / (cm2 s)')

    print(f'\n\nIC (CMB) energy budget:\nNaima model ---> {res1:.3e}\nThis work   ---> {res2:.3e}')

    yval1 = (IC_NIR(energy, *np.repeat(1, 7)))
    res1 = (trapz_loglog(yval1.value/energy.value, energy.value)* yval1.unit).to('eV / (cm2 s)')
    yval2 = (IC_NIR(energy, *fit_params_d3))
    res2 = (trapz_loglog(yval2.value/energy.value, energy.value) * yval2.unit).to('eV / (cm2 s)')

    print(f'\n\nIC (NIR) energy budget:\nNaima model ---> {res1:.3e}\nThis work   ---> {res2:.3e}')

    yval1 = (IC_FIR(energy, *np.repeat(1, 7)))
    res1 = (trapz_loglog(yval1.value/energy.value, energy.value)  * yval1.unit).to('eV / (cm2 s)')
    yval2 = (IC_FIR(energy, *fit_params_d3))
    res2 = (trapz_loglog(yval2.value/energy.value, energy.value)  * yval2.unit).to('eV / (cm2 s)')

    print(f'\n\nIC (FIR) energy budget:\nNaima model ---> {res1:.3e}\nThis work   ---> {res2:.3e}')

    eopts = {'Eemax': 50 * u.PeV, 'Eemin': 0.1 * u.GeV}

    Rpwn = 2.1 * u.pc
    Esy = np.logspace(-7, 9, 100) * u.eV

    def SYN_IC_red(energy, ampl=1, Bmag=1):

        ECBPL = ExponentialCutoffBrokenPowerLaw(amplitude= ampl * 3.699e36 / u.eV, e_0=1 * u.TeV, e_break=fit_params_d3[5]*0.265 * u.TeV, 
                                                alpha_1=fit_params_d3[1]*1.5, alpha_2=fit_params_d3[2]*3.233, e_cutoff=fit_params_d3[4]*1863 * u.TeV, beta=fit_params_d3[3]*2.0,)

        SYN = Synchrotron(ECBPL, B=Bmag*125 * u.uG, Eemax=50 * u.PeV, Eemin=0.1 * u.GeV)
        Lsy = SYN.flux(Esy, distance=0 * u.cm)
        phn_sy = Lsy / (4 * np.pi * Rpwn ** 2 * c) * 2.24
        seed_photon_fields = ['CMB', ['FIR', 70 * u.K, 0.5 * u.eV / u.cm ** 3], ['NIR', 5000 * u.K, 1 * u.eV / u.cm ** 3], ['SSC', Esy, phn_sy],]

        IC = InverseCompton(ECBPL, seed_photon_fields=seed_photon_fields, Eemax=50 * u.PeV, Eemin=0.1 * u.GeV,)    

        return IC.sed(energy, 2 * u.kpc) + SYN.sed(energy, 2 * u.kpc)


    compute =True
    N = 105

    a_arr = np.linspace(fit_params_d3[0]*0, fit_params_d3[0]+6, N)
    b_arr = np.linspace(fit_params_d3[-1]*0, fit_params_d3[-1]+6, N)
    A, B = np.meshgrid(a_arr, b_arr)

    def least_squares(A, B):

        return np.sum(((e2dnde_d3-SYN_IC_red(e_d3, A, B))/e2dnde_err_d3)**2).value

    if compute == True:
        print('\n\nCOMPUTING')
        LS = np.vectorize(least_squares)(A, B)

        np.savetxt(f'data/LS-4.txt', LS.T)

    print('COMPUTED')
    LS = np.loadtxt(f'data/LS-4.txt')

if __name__ == '__main__': 
    globals()[sys.argv[1]]()