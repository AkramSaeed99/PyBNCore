#ifndef SMILE_BULKPROCESSOR_H
#define SMILE_BULKPROCESSOR_H

// {{SMILE_PUBLIC_HEADER}}

class DSL_progress;
class DSL_network;
class DSL_dataset;
struct DSL_datasetMatch;

#include <vector>
#include "intarray.h"

class DSL_bulkProcessor
{
public:
	DSL_bulkProcessor(
		const DSL_dataset& ds, DSL_network& net,
		const std::vector<DSL_datasetMatch>& matching);
		
	int AddOutputNode(int nodeHandle);
	int SetIncludedInputVars(const DSL_intArray &inputVars);

	int Run(DSL_dataset &out, DSL_progress* progress = 0);

private:
	DSL_network &net;
	const DSL_dataset &ds;
	std::vector<DSL_datasetMatch> matching;

	DSL_intArray outputNodes;
	DSL_intArray includedInputs;
};

#endif
