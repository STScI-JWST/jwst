"""
Entry point implementations.
"""


def get_steps():
    """
    Return tuples describing the stpipe.Step subclasses provided
    by this package.  This method is registered with the stpipe.steps
    entry point.

    Returns
    -------
    list of tuple (str, str, bool)
        The first element each tuple is a fully-qualified Step
        subclass name.  The second element is an optional class
        alias.  The third element indicates that the class
        is a subclass of Pipeline.
    """
    # Unit tests ensure that this list is kept in sync with the actual
    # class definitions.  We need to avoid importing jwst.pipeline and
    # jwst.step to keep the CLI snappy.
    return [
        ("jwst.pipeline.Ami3Pipeline", 'calwebb_ami3', True),
        ("jwst.pipeline.Coron3Pipeline", 'calwebb_coron3', True),
        ("jwst.pipeline.DarkPipeline", 'calwebb_dark', True),
        ("jwst.pipeline.Detector1Pipeline", 'calwebb_detector1', True),
        ("jwst.pipeline.GuiderPipeline", 'calwebb_guider', True),
        ("jwst.pipeline.Image2Pipeline", 'calwebb_image2', True),
        ("jwst.pipeline.Image3Pipeline", 'calwebb_image3', True),
        ("jwst.pipeline.Spec2Pipeline", 'calwebb_spec2', True),
        ("jwst.pipeline.Spec3Pipeline", 'calwebb_spec3', True),
        ("jwst.pipeline.Tso3Pipeline", 'calwebb_tso3', True),
        ("jwst.step.AmiAnalyzeStep", None, False),
        ("jwst.step.AmiAverageStep", None, False),
        ("jwst.step.AmiNormalizeStep", None, False),
        ("jwst.step.AssignMTWcsStep", None, False),
        ("jwst.step.AssignWcsStep", None, False),
        ("jwst.step.BackgroundStep", None, False),
        ("jwst.step.BarShadowStep", None, False),
        ("jwst.step.Combine1dStep", None, False),
        ("jwst.step.StackRefsStep", None, False),
        ("jwst.step.AlignRefsStep", None, False),
        ("jwst.step.KlipStep", None, False),
        ("jwst.step.HlspStep", None, False),
        ("jwst.step.CubeBuildStep", None, False),
        ("jwst.step.CubeSkyMatchStep", None, False),
        ("jwst.step.DarkCurrentStep", None, False),
        ("jwst.step.DQInitStep", None, False),
        ("jwst.step.Extract1dStep", None, False),
        ("jwst.step.Extract2dStep", None, False),
        ("jwst.step.FirstFrameStep", None, False),
        ("jwst.step.FlatFieldStep", None, False),
        ("jwst.step.FringeStep", None, False),
        ("jwst.step.GainScaleStep", None, False),
        ("jwst.step.GroupScaleStep", None, False),
        ("jwst.step.GuiderCdsStep", None, False),
        ("jwst.step.ImprintStep", None, False),
        ("jwst.step.IPCStep", None, False),
        ("jwst.step.JumpStep", None, False),
        ("jwst.step.LastFrameStep", None, False),
        ("jwst.step.LinearityStep", None, False),
        ("jwst.step.MasterBackgroundStep", None, False),
        ("jwst.step.MasterBackgroundNrsSlitsStep", None, False),
        ("jwst.step.MRSIMatchStep", None, False),
        ("jwst.step.MSAFlagOpenStep", None, False),
        ("jwst.step.OutlierDetectionStep", None, False),
        ("jwst.step.OutlierDetectionScaledStep", None, False),
        ("jwst.step.OutlierDetectionStackStep", None, False),
        ("jwst.step.PathLossStep", None, False),
        ("jwst.step.PersistenceStep", None, False),
        ("jwst.step.PhotomStep", None, False),
        ("jwst.step.RampFitStep", None, False),
        ("jwst.step.RefPixStep", None, False),
        ("jwst.step.ResampleStep", None, False),
        ("jwst.step.ResampleSpecStep", None, False),
        ("jwst.step.ResetStep", None, False),
        ("jwst.step.RscdStep", None, False),
        ("jwst.step.RSCD_Step", None, False),
        ("jwst.step.SaturationStep", None, False),
        ("jwst.step.SkyMatchStep", None, False),
        ("jwst.step.SourceCatalogStep", None, False),
        ("jwst.step.SourceTypeStep", None, False),
        ("jwst.step.StraylightStep", None, False),
        ("jwst.step.SuperBiasStep", None, False),
        ("jwst.step.TSOPhotometryStep", None, False),
        ("jwst.step.TweakRegStep", None, False),
        ("jwst.step.WavecorrStep", None, False),
        ("jwst.step.WfsCombineStep", 'calwebb_wfs-image3', False),
        ("jwst.step.WfssContamStep", None, False),
        ("jwst.step.WhiteLightStep", None, False),
    ]
