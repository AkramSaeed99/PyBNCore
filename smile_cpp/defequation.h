#ifndef SMILE_DEFEQUATION_H
#define SMILE_DEFEQUATION_H

// {{SMILE_PUBLIC_HEADER}}

#include "nodedef.h"
#include "generalequation.h"
#include <string>
#include <map>

// represents an equation node which can only have
// DSL_equation as its parents.


struct DSL_equationDiscretizationInput
{
	int nodeHandle;
	bool interval;
	double value[2];
};

bool operator==(const DSL_equationDiscretizationInput& lhs, const DSL_equationDiscretizationInput& rhs);

class DSL_equation : public DSL_nodeDef
{
public:
	typedef std::vector<std::pair<std::string, double> > IntervalVector;

	DSL_equation(DSL_network& network, int handle, const char* nodeId);
    DSL_equation(const DSL_nodeXformContext &context);

    int GetType() const { return DSL_EQUATION; }
    const char* GetTypeName() const { return "EQUATION"; }
	
    int AddParent(int theParent);
    int RemoveParent(int theParent);

    const DSL_generalEquation& GetEquation() const { return equation; }
	DSL_expression* GetSolution() { return solution ? solution : Solve(); }
	const DSL_expression* GetSolution() const { return solution ? solution : Solve(); }

	int Evaluate(const std::map<std::string, double>& values, double& value) const;
	int EvaluateConstant(double& value) const;

    int SetEquation(const std::string &eq, int *errPos = NULL, std::string *errMsg = NULL);
    bool ValidateEquation(const std::string &eq, std::vector<std::string> &vars, 
		std::string &errMsg, int *errPos = NULL, bool *isConst = NULL, 
		const char *alternateId = NULL) const;

	int SetTemporalEquations(const std::vector<std::string>& eq, int *errEqIndex = NULL, int* errPos = NULL, std::string* errMsg = NULL);
	bool ValidateTemporalEquations(const std::vector<std::string> &eq, int &errEqIndex, int &errPos, std::string &errMsg, const char* alternateId = NULL) const;
	int GetTemporalEquations(std::vector<std::pair<const DSL_generalEquation *, int> >& eq) const;
	int GetTemporalEquations(std::vector<const DSL_generalEquation*> &eq) const;
	int GetTemporalEquations(std::vector<std::string> &eq) const; 
	bool IsTemporalDeterministic() const;

    int SetBounds(double low, double high);
    void GetBounds(double &low, double &high) const { low = lowBound; high = highBound; }

	int SetNominalRange(double lowNominal, double highNominal);
	void GetNominalRange(double& lowNominal, double& highNominal) const;

	int SetDefaultValue(double defValue);
	bool HasDefaultValue() const;
	void RemoveDefaultValue();
	int GetDefaultValue(double &defValue) const;

	bool HasDiscIntervals() const { return !discIntervals.empty(); }
	void EnsureIntervalsExist();
	const DSL_Dmatrix* GetDiscProbs() const;
	int GetDiscProbInputs(std::vector<DSL_equationDiscretizationInput> &inputs) const;
	void InvalidateDiscProbs() const;
	int SetDiscIntervals(const IntervalVector &intervals) { return SetDiscIntervals(lowBound, highBound, intervals); }
	int SetDiscIntervals(double lo, double hi, const IntervalVector &intervals);
	int SetUniformDiscIntervals(int intervalCount);
	int ClearDiscIntervals();
	const IntervalVector& GetDiscIntervals() const { return discIntervals; }
	int GetDiscInterval(int intervalIndex, double &lo, double &hi) const;
	int Discretize(double x) const;

    void GetIntervalIds(DSL_idArray &states) const;
	void GetIntervalEdges(std::vector<double> &edges) const;
	void GetIntervalEdges(DSL_doubleArray& edges) const;
	void GetIntervalEdges(double* edges) const;

	int OnParentOutcomeAdd(int parentHandle, int thisPosition) { return OnDiscreteParentChange(); }
	int OnParentOutcomeRemove(int parentHandle, int thisPosition) { return OnDiscreteParentChange(); }
	int OnParentOutcomeReorder(int parentHandle, const DSL_intArray &newOrder) { return OnDiscreteParentChange(); }
	int OnParentReorder(const DSL_intArray &newOrder) { return OnDiscreteParentChange(); }

    int ValidateExtFunctions(const DSL_extFunctionContainer &extFxn,int &errFxnIdx, std::string &errMsg) const;
    int PatchExtFunctions(const DSL_extFunctionContainer &extFxn);
	int Patch(DSL_equationPatcher &patcher);
	
    int PrepareForDiscreteChild();

	int CalcDiscProbs(bool useCurrentEvidence, std::vector<std::vector<double> > *samples = NULL) const;

	int AbsorbConstantParents(bool useCurrentEvidence);

	const char* GetId() const;
	const char* GetPeerId(int peerHandle) const;

private:
	DSL_equation(const DSL_equation& src, DSL_network& targetNetwork);
	~DSL_equation();
	DSL_nodeDef* Clone(DSL_network& targetNetwork) const;
	void InitXform(DSL_nodeXformContext &ctx) const;
	DSL_nodeVal* CreateValue() const;
	DSL_nodeVal* CreateValue(const DSL_valXformContext& ctx) const;
	int GetDiscreteOutcomeCount();

	DSL_expression* Solve() const;

	const DSL_Dmatrix* GetParameters() const { return NULL; }
	DSL_Dmatrix* GetWriteableParameters() { return NULL; }

	void BeforeIdChange(const char* oldId, const char* newId);
	void BeforeParentIdChange(int parentHandle, const char *oldId, const char *newId);
	int OnDiscreteParentChange();

	void InvalidateDescendants() const; 
	void InvalidateWithDescendants() const;
	void InvalidateChildrenDiscProbs() const;
	
	const DSL_Dmatrix* GetUnrolledSiblingProbs() const;

    DSL_generalEquation equation;
    double lowBound;
    double highBound;
	double defValue;
	double lowNominal;
	double highNominal;
	mutable DSL_expression* solution;
	
	IntervalVector discIntervals;
	mutable DSL_Dmatrix *discProbs;
	mutable std::vector<DSL_equationDiscretizationInput> *discInputs;
};

#endif
