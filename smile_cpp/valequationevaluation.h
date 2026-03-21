#ifndef SMILE_VALEQUATIONEVALUATION_H
#define SMILE_VALEQUATIONEVALUATION_H

// {{SMILE_PUBLIC_HEADER}}

#include "nodeval.h"
#include "dmatrix.h"
#include <vector>


class DSL_equationEvaluation : public DSL_nodeVal
{
public:
    DSL_equationEvaluation(DSL_network& network, int handle);
    DSL_equationEvaluation(const DSL_valXformContext& ctx);

    int GetType() const { return DSL_EQUATIONEVALUATION; }

    int AddIndexingParent(int parent) { return DSL_WRONG_NODE_TYPE; }
    const DSL_Dmatrix* GetMatrix() const { return &discBeliefs; }
    DSL_Dmatrix* GetWriteableMatrix() { return &discBeliefs; }

    int ValidateAllTemporalEvidence(
        const std::vector<std::pair<int, int> >& d,
        const std::vector<std::pair<int, double> >& c,
        const std::vector<std::pair<int, std::vector<double> > >& v) const;

    int GetMean(double& mean) const;
    int GetStdDev(double& stddev) const;
    int GetTemporalMeanStdDev(DSL_doubleArray& temporalMean, DSL_doubleArray& temporalStdDev) const;
    int GetTemporalMeanStdDev(std::vector<double>& temporalMean, std::vector<double>& temporalStdDev) const;

    const std::vector<double>& GetSamples() const { return samples; }
	double GetSample(int index) const { return samples[index]; }
    const std::vector<int>& GetSampleSliceCounts() const { return sampleSliceCounts; }

    int GetHistogram(double lo, double hi, int binCount, std::vector<int> &histogram) const;

	double GetSampleMean() const;
    double GetSampleStdDev() const;
	void GetStats(double &mean, double &stddev, double &vmin, double &vmax) const;

	bool HasSamplesOutOfBounds() const;

    void SamplingStart(int samplesToReserve = 0);
	void AddSample(double sample) { samples.push_back(sample); }
    int SamplingEnd();

    void SetSamples(const std::vector<double>& allSamples);
    void SetSampleSliceCounts(const std::vector<int>& sliceCounts);

	// used only for exact evaluation
	void SetConstant(double c);

    bool IsDiscretized() const { return !discBeliefs.IsEmpty(); }
    const DSL_Dmatrix& GetDiscBeliefs() const { return discBeliefs; }

    int SetEvidence(int evidence);
    int SetEvidence(double evidence);
    int SetPropagatedEvidence(double evidence);
    int GetEvidence(double& e) const;
    int ClearEvidence();
    int ClearPropagatedEvidence();

    void InitXform(DSL_valXformContext& ctx) const;

    int GetConstantOrEvidence(double& x) const;

private:
    DSL_equationEvaluation(const DSL_equationEvaluation& src, DSL_network& targetNetwork);
    DSL_nodeVal* Clone(DSL_network& targetNetwork) const;

    void CalcStats() const;

    int DoGetTemporalMeanStdDev(double* outMean, double* outStdDev) const;

    double evidence;

    std::vector<double> samples; 
    std::vector<int> sampleSliceCounts;
    
	// if equation is determinisitc (i.e., all its ancestor is deterministic), 
    // sampleMean is its value and sampleStdDev is zero
    mutable double sampleMean;
    mutable double sampleStdDev;
	mutable double sampleMin;
	mutable double sampleMax;

    mutable std::vector<double> temporalMean;
    mutable std::vector<double> temporalStdDev;

	mutable bool statsCalculated;

	DSL_Dmatrix discBeliefs;
};

#endif

