# This example shows how to load data from a SDF file into DeepChem. The data in this SDF file is stored in field "LogP(RRCK)"
import deepchem as dc

featurizer = dc.feat.CircularFingerprint(size=16)
loader = dc.data.SDFLoader(["LogP(RRCK)"], featurizer=featurizer, clean_mols=True)
dataset = loader.featurize("membrane_permeability.sdf")
