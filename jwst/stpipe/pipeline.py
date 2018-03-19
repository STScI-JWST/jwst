# Copyright (C) 2010 Association of Universities for Research in Astronomy(AURA)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#     3. The name of AURA and its representatives may not be used to
#       endorse or promote products derived from this software without
#       specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY AURA ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL AURA BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.
"""
Pipeline

"""
from os.path import dirname, join

from .configobj.configobj import Section

from . import config_parser
from . import Step
from . import crds_client

class Pipeline(Step):
    """
    A Pipeline is a way of combining a number of steps together.
    """

    # Configuration
    spec = """
    output_ext = string(default=".fits")      # Output extension
    suffix = string(default=None)             # Suffix for output file name
    output_use_model = boolean(default=False) # force use `meta.filename` as the output name
    """
    # A set of steps used in the Pipeline.  Should be overridden by
    # the subclass.
    step_defs = {}

    def __init__(self, *args, **kwargs):
        """
        See `Step.__init__` for the parameters.
        """
        Step.__init__(self, *args, **kwargs)

        # Configure all of the steps
        for key, val in self.step_defs.items():
            cfg = self.steps.get(key)
            if cfg is not None:
                new_step = val.from_config_section(
                    cfg, parent=self, name=key,
                    config_file=self.config_file)
            else:
                new_step = val(
                    key, parent=self, config_file=self.config_file,
                    **kwargs.get(key, {}))

            setattr(self, key, new_step)

        self.reference_file_types = self._collect_active_reftypes()

    def _collect_active_reftypes(self):
        """Collect the list of all reftypes for child Steps that are not skipped.
        Overridden reftypes are included but handled normally later by the Pipeline
        version of the get_ref_override() method defined below.
        """
        return [reftype for step in self._unskipped_steps
                for reftype in step.reference_file_types]

    @property
    def _unskipped_steps(self):
        """Return a list of the unskipped Step objects launched by `self`."""
        return [getattr(self, name) for name in self.step_defs
                if not getattr(self, name).skip]

    def get_ref_override(self, reference_file_type):
        """Return any override for `reference_file_type` for any of the steps in
        Pipeline `self`.  OVERRIDES Step.

        Returns
        -------
        override_filepath or None.

        """
        for step in self._unskipped_steps:
            override = step.get_ref_override(reference_file_type)
            if override is not None:
                return override
        return None

    @classmethod
    def merge_config(cls, config, config_file):
        steps = config.get('steps', {})

        # Configure all of the steps
        for key in cls.step_defs:
            cfg = steps.get(key)
            if cfg is not None:
                # If a config_file is specified, load those values and
                # then override them with our values.
                if cfg.get('config_file'):
                    cfg2 = config_parser.load_config_file(
                        join(dirname(config_file or ''), cfg.get('config_file')))
                    del cfg['config_file']
                    config_parser.merge_config(cfg2, cfg)
                    steps[key] = cfg2

        return config

    @classmethod
    def load_spec_file(cls, preserve_comments=False):
        spec = config_parser.get_merged_spec_file(
            cls, preserve_comments=preserve_comments)

        spec['steps'] = Section(spec, spec.depth + 1, spec.main, name="steps")
        steps = spec['steps']
        for key, val in cls.step_defs.items():
            if not issubclass(val, Step):
                raise TypeError(
                    "Entry {0!r} in step_defs is not a Step subclass"
                    .format(key))
            stepspec = val.load_spec_file(preserve_comments=preserve_comments)
            steps[key] = Section(steps, steps.depth + 1, steps.main, name=key)

            config_parser.merge_config(steps[key], stepspec)

            # Also add a key that can be used to specify an external
            # config_file
            step = spec['steps'][key]
            step['config_file'] = 'string(default=None)'
            step['name'] = "string(default='')"
            step['class'] = "string(default='')"

        return spec

    def set_input_filename(self, path):
        self._input_filename = path
        for key in self.step_defs:
            getattr(self, key).set_input_filename(path)

    def _precache_references(self, input_file):
        """
        Precache all of the expected reference files before the Step's
        process method is called.

        Handles opening `input_file` as a model if it is a filename.

        input_file:  filename, model container, or model

        returns:  None
        """
        from .. import datamodels
        try:
            with datamodels.open(input_file) as model:
                self._precache_references_opened(model)
        except (ValueError, TypeError, IOError):
            self.log.info(
                'First argument {0} does not appear to be a '
                'model'.format(input_file))

    def _precache_references_opened(self, model_or_container):
        """Pre-fetches references for `model_or_container`.

        Handles recursive pre-fetches for any models inside a container,
        or just a single model.

        Assumes model_or_container is an open model or container object,
        not a filename.

        No garbage collection.
        """
        if self._is_container(model_or_container):
            # recurse on each contained model
            for contained_model in model_or_container:
                self._precache_references_opened(contained_model)
        else:
            # precache a single model object
            self._precache_references_impl(model_or_container)

    def _precache_references_impl(self, model):
        """Given open data `model`,  determine and cache reference files for
        any reference types which are not overridden on the command line.

        Verify that all CRDS and overridden reference files are readable.

        model:  An open Model object;  not a filename, ModelContainer, etc.
        """
        ovr_refs = {
            reftype: self.get_ref_override(reftype)
            for reftype in self.reference_file_types
            if self.get_ref_override(reftype) is not None
            }

        fetch_types = sorted(set(self.reference_file_types) - set(ovr_refs.keys()))

        crds_refs = crds_client.get_multiple_reference_paths(model, fetch_types)

        ref_path_map = dict(list(crds_refs.items()) + list(ovr_refs.items()))

        for (reftype, refpath) in sorted(ref_path_map.items()):
            how = "Override" if reftype in ovr_refs else "Prefetch"
            self.log.info("{0} for {1} reference file is '{2}'.".format(how, reftype.upper(), refpath))
            crds_client.check_reference_open(refpath)

    @classmethod
    def _is_container(cls, input_file):
        """Return True IFF `input_file` is a ModelContainer or successfully
        loads as an association.
        """
        from ..associations import load_asn
        from .. import datamodels
        if isinstance(input_file, datamodels.ModelContainer):
            return True
        try:
            with open(input_file, 'r') as input_file_fh:
                _asn = load_asn(input_file_fh)
        except Exception:
            return False
        return True

