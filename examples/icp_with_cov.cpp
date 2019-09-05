

#include "pointmatcher/PointMatcher.h"
#include "pointmatcher/Bibliography.h"

#include "boost/filesystem.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;

typedef PointMatcher<float> PM;
typedef PM::DataPoints DP;
typedef PM::Parameters Parameters;
typedef PointMatcherSupport::CurrentBibliography CurrentBibliography;

void listModules();
int validateArgs(const int argc, const char *argv[],
				 bool& isVerbose,
				 string& configFile,
				 string& outputBaseFile,
				 string& outputBaseFileCov,
				 string& initTranslation, string& initRotation);
PM::TransformationParameters parseTranslation(string& translation,
											  const int cloudDimension);
PM::TransformationParameters parseRotation(string& rotation,
										   const int cloudDimension);
// Helper functions
void usage(const char *argv[]);

/**
  * Code example for ICP taking 2 points clouds (2D or 3D) relatively close
  * and computing the transformation between them and Censi covariance.
  *
  * This code is more complete than icp_simple. It can load parameter files and
  * has more options.
  */

int main(int argc, const char *argv[])
{

	bool isVerbose = false;

	string configFile;
	string outputBaseFile("");
	string outputBaseFileCov("");
	string initTranslation("0,0,0");
	string initRotation("1,0,0;0,1,0;0,0,1");
	const int ret = validateArgs(argc, argv, isVerbose, configFile,
								 outputBaseFile, outputBaseFileCov, initTranslation, initRotation);
	if (ret != 0)
	{
		return ret;
	}
	const char *refFile(argv[argc-2]);
	const char *dataFile(argv[argc-1]);

	// Load point clouds
	const DP ref(DP::load(refFile));
	const DP data(DP::load(dataFile));

	// Create the default ICP algorithm
	PM::ICP icp;

	if (configFile.empty())
	{
		// See the implementation of setDefault() to create a custom ICP algorithm
		icp.setDefault();
	}
	else
	{
		// load YAML config
		ifstream ifs(configFile.c_str());
		if (!ifs.good())
		{
			cerr << "Cannot open config file " << configFile << ", usage:"; usage(argv); exit(1);
		}
		icp.loadFromYaml(ifs);
	}


	int cloudDimension = ref.getEuclideanDim();
	if (!(cloudDimension == 2 || cloudDimension == 3))
	{
		cerr << "Invalid input point clouds dimension" << endl;
		exit(1);
	}

	PM::TransformationParameters translation =
			parseTranslation(initTranslation, cloudDimension);
	PM::TransformationParameters rotation =
			parseRotation(initRotation, cloudDimension);
	PM::TransformationParameters initTransfo = translation*rotation;

	std::shared_ptr<PM::Transformation> rigidTrans;
	rigidTrans = PM::get().REG(Transformation).create("RigidTransformation");

	if (!rigidTrans->checkParameters(initTransfo)) {
		cerr << endl
			 << "Initial transformation is not rigid, identiy will be used"
			 << endl;
		initTransfo = PM::TransformationParameters::Identity(
					cloudDimension+1,cloudDimension+1);
	}


	// Compute the transformation to express data in ref
	PM::TransformationParameters T = icp(data, ref, initTransfo);
	if(isVerbose)
		cout << "match ratio: " << icp.errorMinimizer->getWeightedPointUsedRatio() << endl;

	ofstream transfoFile;
	string completeFileName = outputBaseFile;
	transfoFile.open(completeFileName.c_str());
	if(transfoFile.is_open()) {
    transfoFile << T << endl;
    transfoFile << initTransfo << endl;
		transfoFile.close();
	} else {
		cerr << "Unable to write the complete transformation file\n" << endl;
	}

	PM::Matrix censi_cov, bonnabel_cov;
	icp.errorMinimizer->getCovariance(censi_cov, bonnabel_cov);
	completeFileName = outputBaseFileCov;
	transfoFile.open(completeFileName.c_str());
		if(transfoFile.is_open()) {
			//transfoFile << T*initTransfo << endl;
			transfoFile << censi_cov << endl;
			transfoFile << bonnabel_cov << endl;
			transfoFile.close();
		} else {
			cerr << "Unable to write the complete covariance file\n" << endl;
		}
	return 0;
}

// The following code allows to dump all existing modules
template<typename R>
void dumpRegistrar(const PM& pm, const R& registrar, const std::string& name,
				   CurrentBibliography& bib)
{
	cout << "* " << name << " *\n" << endl;
	for (BOOST_AUTO(it, registrar.begin()); it != registrar.end(); ++it)
	{
		cout << it->first << endl;
		cout << getAndReplaceBibEntries(it->second->description(), bib) << endl;
		cout << it->second->availableParameters() << endl;
	}
	cout << endl;
}

#define DUMP_REGISTRAR_CONTENT(pm, name, bib) \
	dumpRegistrar(pm, pm.REG(name), # name, bib);

void listModules()
{
	CurrentBibliography bib;

	DUMP_REGISTRAR_CONTENT(PM::get(), Transformation, bib)
			DUMP_REGISTRAR_CONTENT(PM::get(), DataPointsFilter, bib)
			DUMP_REGISTRAR_CONTENT(PM::get(), Matcher, bib)
			DUMP_REGISTRAR_CONTENT(PM::get(), OutlierFilter, bib)
			DUMP_REGISTRAR_CONTENT(PM::get(), ErrorMinimizer, bib)
			DUMP_REGISTRAR_CONTENT(PM::get(), TransformationChecker, bib)
			DUMP_REGISTRAR_CONTENT(PM::get(), Inspector, bib)
			DUMP_REGISTRAR_CONTENT(PM::get(), Logger, bib)

			cout << "* Bibliography *" << endl << endl;
	bib.dump(cout);
}

// Make sure that the command arguments make sense
int validateArgs(const int argc, const char *argv[],
				 bool& isVerbose,
				 string& configFile,
				 string& outputBaseFile,
				 string& outputBaseFileCov,
				 string& initTranslation, string& initRotation)
{
	if (argc == 1)
	{
		cerr << "Not enough arguments, usage:";
		usage(argv);
		return 1;
	}
	else if (argc == 2)
	{
		if (string(argv[1]) == "-l")
		{
			listModules();
			return -1; // we use -1 to say that we wish to quit but in a normal way
		}
		else
		{
			cerr << "Wrong option, usage:";
			usage(argv);
			return 2;
		}
	}

	const int endOpt(argc - 2);
	for (int i = 1; i < endOpt; i += 2)
	{
		const string opt(argv[i]);
		if (opt == "--verbose" || opt == "-v") {
			isVerbose = true;
			i --;
			continue;
		}


	   if (opt == "--config") {
			configFile = argv[i+1];
		}
		else if (opt == "--output") {
			outputBaseFile = argv[i+1];
		}
		else if (opt == "--output_cov") {
			outputBaseFileCov = argv[i+1];
		}
		else if (opt == "--initTranslation") {
			initTranslation = argv[i+1];
		}
		else if (opt == "--initRotation") {
			initRotation = argv[i+1];
		}
		else
		{
			cerr << "Unknown option " << opt << ", usage:"; usage(argv); exit(1);
		}
	}
	return 0;
}

PM::TransformationParameters parseTranslation(string& translation,
											  const int cloudDimension) {
	PM::TransformationParameters parsedTranslation;
	parsedTranslation = PM::TransformationParameters::Identity(
				cloudDimension+1,cloudDimension+1);

	translation.erase(std::remove(translation.begin(), translation.end(), '['),
					  translation.end());
	translation.erase(std::remove(translation.begin(), translation.end(), ']'),
					  translation.end());
	std::replace( translation.begin(), translation.end(), ',', ' ');
	std::replace( translation.begin(), translation.end(), ';', ' ');

	float translationValues[3] = {0};
	stringstream translationStringStream(translation);
	for( int i = 0; i < cloudDimension; i++) {
		if(!(translationStringStream >> translationValues[i])) {
			cerr << "An error occured while trying to parse the initial "
				 << "translation." << endl
				 << "No initial translation will be used" << endl;
			return parsedTranslation;
		}
	}
	float extraOutput = 0;
	if((translationStringStream >> extraOutput)) {
		cerr << "Wrong initial translation size" << endl
			 << "No initial translation will be used" << endl;
		return parsedTranslation;
	}

	for( int i = 0; i < cloudDimension; i++) {
		parsedTranslation(i,cloudDimension) = translationValues[i];
	}

	return parsedTranslation;
}

PM::TransformationParameters parseRotation(string &rotation,
										   const int cloudDimension){
	PM::TransformationParameters parsedRotation;
	parsedRotation = PM::TransformationParameters::Identity(
				cloudDimension+1,cloudDimension+1);

	rotation.erase(std::remove(rotation.begin(), rotation.end(), '['),
				   rotation.end());
	rotation.erase(std::remove(rotation.begin(), rotation.end(), ']'),
				   rotation.end());
	std::replace( rotation.begin(), rotation.end(), ',', ' ');
	std::replace( rotation.begin(), rotation.end(), ';', ' ');

	float rotationMatrix[9] = {0};
	stringstream rotationStringStream(rotation);
	for( int i = 0; i < cloudDimension*cloudDimension; i++) {
		if(!(rotationStringStream >> rotationMatrix[i])) {
			cerr << "An error occured while trying to parse the initial "
				 << "rotation." << endl
				 << "No initial rotation will be used" << endl;
			return parsedRotation;
		}
	}
	float extraOutput = 0;
	if((rotationStringStream >> extraOutput)) {
		cerr << "Wrong initial rotation size" << endl
			 << "No initial rotation will be used" << endl;
		return parsedRotation;
	}

	for( int i = 0; i < cloudDimension*cloudDimension; i++) {
		parsedRotation(i/cloudDimension,i%cloudDimension) = rotationMatrix[i];
	}

	return parsedRotation;
}

// Dump command-line help
void usage(const char *argv[])
{
	cerr << endl << endl;
	cerr << "* To list modules:" << endl;
	cerr << "  " << argv[0] << " -l" << endl;
	cerr << endl;
	cerr << "* To run ICP:" << endl;
	cerr << "  " << argv[0] << " [OPTIONS] reference.csv reading.csv" << endl;
	cerr << endl;
	cerr << "OPTIONS can be a combination of:" << endl;
	cerr << "-v,--verbose               Be more verbose (info logging to stdout)" << endl;
	cerr << "--config YAML_CONFIG_FILE  Load the config from a YAML file (default: default parameters)" << endl;
	cerr << "--output BASEFILENAME      Name of output files (default: "")" << endl;
	cerr << "--output_cov BASEFILENAME      Name of output cov files (default: "")" << endl;
	cerr << "--initTranslation [x,y,z]  Add an initial 3D translation before applying ICP (default: 0,0,0)" << endl;
	cerr << "--initTranslation [x,y]    Add an initial 2D translation before applying ICP (default: 0,0)" << endl;
	cerr << "--initRotation [r00,r01,r02,r10,r11,r12,r20,r21,r22]" << endl;
	cerr << "                           Add an initial 3D rotation before applying ICP (default: 1,0,0,0,1,0,0,0,1)" << endl;
	cerr << "--initRotation [r00,r01,r10,r11]" << endl;
	cerr << "                           Add an initial 2D rotation before applying ICP (default: 1,0,0,1)" << endl;
	cerr << endl;
}

