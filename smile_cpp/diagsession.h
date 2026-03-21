#ifndef SMILE_DIAGSESSION_H
#define SMILE_DIAGSESSION_H

// {{SMILE_PUBLIC_HEADER}}

#include <vector>
#include <algorithm>
#include "intarray.h"

class DSL_network;
class DSL_discDef;
class DSL_diagCalculator;

// Flags defining the entropy-based multiple fault diagnosis
#define DSL_DIAG_MARGINAL		1
#define DSL_DIAG_INDEPENDENCE	2
#define DSL_DIAG_DEPENDENCE		4
#define DSL_DIAG_PURSUE_ATLEAST_ONE_COMB	8
#define DSL_DIAG_PURSUE_ONLY_ONE_COMB		16
#define DSL_DIAG_PURSUE_ONLY_ALL_COMB		32
#define DSL_DIAG_MARGINAL_STRENGTH1	64
#define DSL_DIAG_MARGINAL_STRENGTH2	128

// multi-fault diagnosis algorithms
#define DSL_DIAG_MULTI_MARGINAL1 (DSL_DIAG_MARGINAL | DSL_DIAG_MARGINAL_STRENGTH1)
#define DSL_DIAG_MULTI_MARGINAL2 (DSL_DIAG_MARGINAL | DSL_DIAG_MARGINAL_STRENGTH2)
#define DSL_DIAG_MULTI_DEPENDENCE_ATLEAST1 (DSL_DIAG_DEPENDENCE | DSL_DIAG_PURSUE_ATLEAST_ONE_COMB)
#define DSL_DIAG_MULTI_DEPENDENCE_ONLY1 (DSL_DIAG_DEPENDENCE | DSL_DIAG_PURSUE_ONLY_ONE_COMB)
#define DSL_DIAG_MULTI_DEPENDENCE_ALL (DSL_DIAG_DEPENDENCE | DSL_DIAG_PURSUE_ONLY_ALL_COMB)
#define DSL_DIAG_MULTI_INDEPENDENCE_ATLEAST1 (DSL_DIAG_INDEPENDENCE | DSL_DIAG_PURSUE_ATLEAST_ONE_COMB)
#define DSL_DIAG_MULTI_INDEPENDENCE_ONLY1 (DSL_DIAG_INDEPENDENCE | DSL_DIAG_PURSUE_ONLY_ONE_COMB)
#define DSL_DIAG_MULTI_INDEPENDENCE_ALL (DSL_DIAG_INDEPENDENCE | DSL_DIAG_PURSUE_ONLY_ALL_COMB)

#define DSL_DIAG_MULTI_DISTANCE_BASE 256
#define DSL_DIAG_MULTI_MAX_PROB_CHANGE (DSL_DIAG_MULTI_DISTANCE_BASE | 1) 
#define DSL_DIAG_MULTI_L2_NORMALIZED_DISTANCE (DSL_DIAG_MULTI_DISTANCE_BASE | 2)
#define DSL_DIAG_MULTI_COSINE_DISTANCE (DSL_DIAG_MULTI_DISTANCE_BASE | 3)
#define DSL_DIAG_MULTI_CITYBLOCK_DISTANCE (DSL_DIAG_MULTI_DISTANCE_BASE | 4)
#define DSL_DIAG_MULTI_AVG_L2_CITY_DISTANCE (DSL_DIAG_MULTI_DISTANCE_BASE | 5)

// single-fault diagnosis algorithms
#define DSL_DIAG_SINGLE_NORM_CROSSENTROPY 512
#define DSL_DIAG_SINGLE_CROSSENTROPY 1024
#define DSL_DIAG_SINGLE_PROB_CHANGE 2048

#define DSL_DIAG_DEFAULT DSL_DIAG_MULTI_INDEPENDENCE_ATLEAST1

#define DSL_DIAG_CONT_FAULT_LOW (-1)
#define DSL_DIAG_CONT_FAULT_HIGH 1

struct DSL_diagTestInfo
{
	DSL_diagTestInfo(int testHandle = 0);
	int test;
	double entropy;
	double cost;
	double strength;
	int observationPriorStartIndex;
	int faultPosteriorsStartIndex;
	int outcomeVoiStartIndex;
};

struct DSL_diagFaultState
{
	int node;
	int state;
};

// used only by deprecated CalculateRankedFaults
struct DSL_diagFaultInfo
{
	int nodeHandle;
	int nodeState;
	int faultHandle;
	double posterior;
};

bool operator<(const DSL_diagFaultState& lhs, const DSL_diagFaultState& rhs);
bool operator<(const DSL_diagFaultInfo& lhs, const DSL_diagFaultInfo& rhs);

class DSL_diagSession
{
public:
	DSL_diagSession(DSL_network &diagNetwork);  
	~DSL_diagSession();

	DSL_network& GetNetwork() { return net; }
	
	int RestartDiagnosis();
	
	int UpdateFaultBeliefs();
	int UpdateTestStrengths();

	int GetSingleFaultAlgorithm() const { return singleFaultAlg; }
	int GetMultiFaultAlgorithm() const { return multiFaultAlg; }

	int SetSingleFaultAlgorithm(int algId);
	int SetMultiFaultAlgorithm(int algId);

	// all faults (node/outcome pairs) in the model
	const std::vector<DSL_diagFaultState>& GetFaults() const { return faults; }

	int GetFaultProbability(int faultIndex, double &probability) const;

	int GetPursuedFault() const;
	const DSL_intArray& GetPursuedFaults() const { return pursuedFaults; }
	int SetPursuedFault(int faultIndex);
	int SetPursuedFaults(const DSL_intArray& faultIndices);
	int AddPursuedFault(int faultIndex);
	int DeletePursuedFault(int faultIndex);

	// returns the node handles of unperformed tests
	const DSL_intArray& GetUnperformedTests() const { return unperformedTests; }
	// returns the array of tests statistics. This array is paired with the
	// one returned by GetUnperformedTests. 
	const std::vector<DSL_diagTestInfo>& GetTestStatistics() const { return testStatistics; }
	const std::vector<DSL_diagTestInfo>& GetPastTestStatistics() const { return pastTestStatistics; }

	bool IsDSepEnabled() const { return dsepEnabled; }
	void EnableDSep(bool enable) { dsepEnabled = enable; }

	bool AreQuickTestsEnabled() const { return quickTestsEnabled; }
	void EnableQuickTests(bool enable) { quickTestsEnabled = enable; }

	bool ArePastTestStatsEnabled() const { return pastTestStatsEnabled; }
	void EnablePastTestStats(bool enable) { pastTestStatsEnabled = enable; }

	const std::vector<double>& GetDetailedStats() const { return allTestOutcomeStats; }

	// both CalculateRankedFaults overloads are deprecated
	void CalculateRankedFaults(std::vector<DSL_diagFaultInfo> &here, double lower, double upper);
	void CalculateRankedFaults(std::vector<DSL_diagFaultInfo> &here, bool openSet = false);

	// sets the ratio in [theNet] as a user-defined property
	void SetEntropyCostRatio(double ratio, double maxRatio);
	double GetEntropyCostRatio() const;
	double GetMaxEntropyCostRatio() const;

	bool MandatoriesInstantiated() const;

	int InstantiateObservation(int nodeHandle, int outcomeIndex);
	int InstantiateObservation(int nodeHandle, double value);
	int ReleaseObservation(int nodeHandle);

	void SetDefaultStates();

	// returns the handle of the most likely fault, assuming that (assummes 
	// that UpdateFaultBeliefs was called already
	int FindMostLikelyFault();
	int FindFault(int faultNode, int faultState) const;

	void AppendTestStatistics(const DSL_diagTestInfo& ti);
	void AppendPastTestStatistics(const DSL_diagTestInfo& ti);
	void AppendTestOutcomeStats(const double* p, int count);

private:
	// was previously public, use UpdateTestStrengths(void) instead,
	// specify algorithms with Set[Single|Multi]FaultAlgorithm
	int ComputeTestStrengths(int flags = DSL_DIAG_DEFAULT);

	void InitFaults();
	void InitUnperformedObservations();
	bool IsDiscretizationRequired() const { return NULL != discNet; }
	int ComputeDiscretizedTestStrengths(int flags = DSL_DIAG_DEFAULT);
	void SetDiscretizedPursuedFaults();
	void SetDiscretizedObservations();
	int UpdateDiscretizedBeliefs();
	bool IsFaultIndexValid(int faultIndex) const;
	double GetFromProperty(const char *propertyName, double defaultValue) const;
	void DoCalcRankedFaults(std::vector<DSL_diagFaultInfo> &here, double lower, double upper, bool openSet );
	void ClearTestStats();
	void GetDiscreteFaultStates(int faultNode, int faultState, DSL_intArray& discreteStates) const;

	DSL_network &net;
	DSL_diagCalculator *calc;

	// contains all ranked observations not observed yet
	DSL_intArray unperformedTests;
	std::vector<DSL_diagTestInfo> testStatistics;
	std::vector<DSL_diagTestInfo> pastTestStatistics;

	std::vector<DSL_diagFaultState> faults;
	DSL_intArray pursuedFaults; // indices in [faults]
	std::vector<double> allTestOutcomeStats;

	int singleFaultAlg;
	int multiFaultAlg;

	bool dsepEnabled;
	bool quickTestsEnabled;
	bool pastTestStatsEnabled;

	bool restoreImmediateUpdate;

	DSL_network* discNet;
	DSL_diagSession* discSession;
};

#endif
