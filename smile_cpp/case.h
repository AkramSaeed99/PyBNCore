#ifndef SMILE_SIMPLECASE_H
#define SMILE_SIMPLECASE_H

// {{SMILE_PUBLIC_HEADER}}

#include <string>
#include <utility>
#include <vector>

class DSL_intArray;
class DSL_network;

class DSL_case
{
public:
    DSL_case(DSL_network& n) : network(n), created(0), lastModified(0) {}
    DSL_case(DSL_network& n, const DSL_case & src);

    // name is unique, but can contain spaces, etc.
    int SetName(const std::string& name);
    const std::string& GetName() const { return name; }
	
    void SetDescription(const std::string& d) { description = d; }
    const std::string& GetDescription() const { return description; }
    void SetCategory(const std::string& c) { category = c; }
    const std::string& GetCategory() const { return category; }

    time_t GetCreated() const { return created; }
    void SetCreated(time_t t) { created = t; }
    time_t GetLastModified() const { return lastModified; }
    void SetLastModified(time_t t) { lastModified = t; } 

    // methods for evidence
    int GetNumberOfEvidence() const { return int(evidence.size()); }
    int GetHandle(int index) const;
    int FindNode(int handle) const { return HandleToIndex(handle); }
    void GetEvidenceNodes(std::vector<int>& evidenceNodeHandles) const;

    bool IsDiscreteEvidence(int handle, int slice = 0) const;
    bool IsContinuousEvidence(int handle, int slice = 0) const;
    bool IsVirtualEvidence(int handle, int slice = 0) const;

    int RemoveEvidence(int index);
    int RemoveEvidenceByHandle(int handle) { return RemoveEvidence(HandleToIndex(handle)); }

    int AddEvidence(int handle, int outcome);
    int AddEvidence(int handle, double value);
	int AddEvidence(int handle, const std::vector<double> &virtualEvidence);
    int AddTemporalEvidence(int handle,
        const std::vector<std::pair<int, int> >& discEvidence,
        const std::vector<std::pair<int, double> >& contEvidence,
        const std::vector<std::pair<int, std::vector<double> > >& virtEvidence);

    int AppendTemporalEvidence(int handle, int slice, int outcome);
    int AppendTemporalEvidence(int handle, int slice, double value);
    int AppendTemporalEvidence(int handle, int slice, const std::vector<double> &virtEvidence);

    int SetEvidence(int index, int outcome);
    int SetEvidence(int index, double value);
    int SetEvidence(int index, const std::vector<double>& virtualEvidence);
    int SetTemporalEvidence(int index,
        const std::vector<std::pair<int, int> >& discEvidence,
        const std::vector<std::pair<int, double> >& contEvidence,
        const std::vector<std::pair<int, std::vector<double> > >& virtEvidence);

    int SetEvidenceByHandle(int handle, int outcome);
    int SetEvidenceByHandle(int handle, double value);
    int SetEvidenceByHandle(int handle, const std::vector<double>& virtualEvidence);
    int SetTemporalEvidenceByHandle(int handle,
        const std::vector<std::pair<int, int> >& discEvidence,
        const std::vector<std::pair<int, double> >& contEvidence,
        const std::vector<std::pair<int, std::vector<double> > >& virtEvidence);

    int GetEvidence(int index, int &handle, int &outcome) const;
    int GetEvidence(int index, int &handle, double &value) const;
	int GetEvidence(int index, int &handle, std::vector<double> &virtualEvidence) const;
    int GetTemporalEvidence(int index, int& handle,
        std::vector<std::pair<int, int> >& discEvidence,
        std::vector<std::pair<int, double> >& contEvidence,
        std::vector<std::pair<int, std::vector<double> > >& virtEvidence) const;

    int GetEvidenceByHandle(int handle, int &outcome) const { int unused; return GetEvidence(HandleToIndex(handle), unused, outcome); }
    int GetEvidenceByHandle(int handle, double &value) const { int unused; return GetEvidence(HandleToIndex(handle), unused, value); }
	int GetEvidenceByHandle(int handle, std::vector<double> &virtualEvidence) const { int unused; return GetEvidence(HandleToIndex(handle), unused, virtualEvidence); }
    int GetTemporalEvidenceByHandle(int handle,
        std::vector<std::pair<int, int> >& discEvidence,
        std::vector<std::pair<int, double> >& contEvidence,
        std::vector<std::pair<int, std::vector<double> > >& virtEvidence) const
    {
        int unused; return GetTemporalEvidence(HandleToIndex(handle), unused, discEvidence, contEvidence, virtEvidence); 
    }

    int GetEvidenceSlices(int index, std::vector<int>& discreteSlices, std::vector<int> continuousSlices, std::vector<int> virtualSlices);

    // target methods
    int AddTarget(int handle);
	void RemoveTarget(int handle);
	int FindTarget(int handle) const;
    int GetTarget(int index) const;
    bool IsTarget(int handle) const;
    int GetNumberOfTargets() const { return int(targets.size()); }
    void SetTargets(const std::vector<int> &nodes) { targets = nodes; }

    // case <-> network
    void CaseToNetwork();
    void NetworkToCase();

    // methods that are called when changes in the network occur
    void OnTypeChanged(int handle, int prevType);
    void OnTemporalTypeChanged(int handle, int prevTemporalType);
    void OnNodeDeleted(int handle);
    void OnOutcomeAdded(int handle, int outcomeIndex);
    void OnOutcomeDeleted(int handle, int outcomeIndex);
    void OnReorderOutcomes(int handle, const DSL_intArray &newOrder);
    
private:
    DSL_network& network;      
    
    // name of the case, is unique in a case manager
    std::string name;         
    std::string description;
    std::string  category;
    time_t created;
    time_t lastModified;

    std::vector<int> targets;      // handles of target nodes

    struct SubItem
    {
        enum { typeDiscrete, typeContinuous, typeVirtual };
        SubItem() : type(-1), discrete(-1), continuous(0), slice(0) {}
        void SetDiscrete(int ev) { type = typeDiscrete; discrete = ev; }
        void SetContinuous(double ev) { type = typeContinuous; continuous = ev; }
        void SetVirtual(const std::vector<double>& ev) { type = typeVirtual; virt = ev; }
        
        static bool PredSliceLess(const SubItem& lhs, const SubItem& rhs) { return lhs.slice < rhs.slice; }
        static bool PredNonContinuous(const SubItem& si) { return si.type != typeContinuous; }

        int type;
        int discrete;
        double continuous;
        std::vector<double> virt;
        int slice;
    };

    struct PredSubItemDiscreteOutcome
    {
        bool operator()(const SubItem& si) 
        { 
            return si.type == SubItem::typeDiscrete && si.discrete == outcome; 
        }
        int outcome;
    };

    struct Item
    {
        Item(int h) : handle(h) {}
        int handle;
        std::vector<SubItem> ev;
        SubItem& EnsureZeroSliceExists();
        SubItem& EnsureSliceExists(int slice);
        const SubItem* GetSlice(int slice) const;
        int SetDiscrete(int ev);
        int SetContinuous(double ev);
        int SetVirtual(const std::vector<double>& ev);
        int SetTemporal(
            const std::vector<std::pair<int, int> >& discEvidence,
            const std::vector<std::pair<int, double> >& contEvidence,
            const std::vector<std::pair<int, std::vector<double> > >& virtEvidence);
    };

    std::vector<Item> evidence;

    bool ValidateItemIndex(int index) const;
    bool ValidateOutcomeIndex(int handle, int outcomeIndex) const;
    int HandleToIndex(int handle) const;
    int GetEvidenceHelper(int index, int expectedType, int& handle, const SubItem* &si) const;
    
    int AddEvidenceProlog(int handle);
    int AppendTemporalEvidenceProlog(int handle, int slice);
    Item& EnsureNodeItemExists(int handle);

    bool SupportsContinuousEvidence(int handle) const;
    bool SupportsVirtualEvidence(int handle, const std::vector<double> &vev) const;
    bool IsPlateNode(int handle) const;
    bool IsEvidenceOfType(int handle, int slice, int expectedType) const;
};

#endif
