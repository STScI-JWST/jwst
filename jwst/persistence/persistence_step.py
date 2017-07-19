#! /usr/bin/env python

from ..stpipe import Step, cmdline
from .. import datamodels
from . import persistence

class PersistenceStep(Step):
    """
    PersistenceStep: Correct a science image for persistence.
    """

    spec = """
        # `input_trapsfilled` is the name of the most recent trapsfilled
        # file for the current detector.
        # Pixels that have received a persistence correction greater than
        # or equal to `flag_pers_cutoff` DN will be flagged in the pixeldq
        # extension of the output (rootname_persistence.fits) file.
        # if `save_persistence` is True, the persistence that was
        # subtracted (group by group, integration by integration) will be
        # written to an output file with suffix "_output_pers".
        input_trapsfilled = string(default="")
        flag_pers_cutoff = float(default=40.)
        save_persistence = boolean(default=False)
    """

    reference_file_types = ["trapdensity", "trappars", "persat"]

    def process(self, input):

        if self.input_trapsfilled is not None:
            if (self.input_trapsfilled == "None" or
                len(self.input_trapsfilled) == 0):
                self.input_trapsfilled = None

        output_obj = datamodels.open(input).copy()

        self.trap_density_filename = self.get_reference_file(output_obj,
                                                             "trapdensity")
        self.trappars_filename = self.get_reference_file(output_obj,
                                                         "trappars")
        self.persat_filename = self.get_reference_file(output_obj, "persat")

        # Is any reference file missing?
        missing = False
        missing_reftypes = []
        if self.persat_filename == "N/A":
            missing = True
            missing_reftypes.append("PERSAT")
        if self.trap_density_filename == "N/A":
            missing = True
            missing_reftypes.append("TRAPDENSITY")
        if self.trappars_filename == "N/A":
            missing = True
            missing_reftypes.append("TRAPPARS")
        if missing:
            if len(missing_reftypes) == 1:
                msg = "Missing reference file type:  " + missing_reftypes[0]
            else:
                msg = "Missing reference file types: "
                for name in missing_reftypes:
                    msg += (" " + name)
            self.log.warning("%s", msg)
            output_obj.meta.cal_step.persistence = "SKIPPED"
            return output_obj

        if self.input_trapsfilled is None:
            traps_filled_model = None
        else:
            traps_filled_model = datamodels.TrapsFilledModel(
                                        self.input_trapsfilled)
        trap_density_model = datamodels.TrapDensityModel(
                                self.trap_density_filename)
        trappars_model = datamodels.TrapParsModel(self.trappars_filename)
        persat_model = datamodels.PersistenceSatModel(self.persat_filename)

        pers_a = persistence.DataSet(output_obj, traps_filled_model,
                                     self.flag_pers_cutoff,
                                     self.save_persistence,
                                     trap_density_model, trappars_model,
                                     persat_model)
        (output_obj, traps_filled, output_pers, skipped) = pers_a.do_all()
        if skipped:
            output_obj.meta.cal_step.persistence = 'SKIPPED'
        else:
            output_obj.meta.cal_step.persistence = 'COMPLETE'

        if traps_filled_model is not None:      # input traps_filled
            traps_filled_model.close()
        if traps_filled is not None:            # output traps_filled
            # Save the traps_filled image, using the input file name but
            # with suffix 'trapsfilled'.
            self.save_model(traps_filled, 'trapsfilled')
            traps_filled.close()

        if output_pers is not None:             # output file of persistence
            self.save_model(output_pers, 'output_pers')
            output_pers.close()

        # Close reference files.
        trap_density_model.close()
        trappars_model.close()
        persat_model.close()

        return output_obj


if __name__ == '__main__':
    cmdline.step_script(persistence_step)
