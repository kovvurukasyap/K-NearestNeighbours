#ifndef TREEINDEX_H
#define TREEINDEX_H

#include "VectorDataset.h"
#include "DataVector.h"
#include <vector>

class TreeIndex
{
protected:
    TreeIndex() {}
    virtual ~TreeIndex() {}

public:
    static TreeIndex &GetInstance();
    virtual void AddData(const DataVector &data) = 0;
    virtual void RemoveData(const DataVector &data) = 0;
    virtual void MakeTree() = 0;
    virtual std::vector<DataVector> Search(const DataVector &test_point, int k) = 0;
};

class KDTreeIndex : public TreeIndex
{
public:
    static KDTreeIndex &GetInstance();

private:
    KDTreeIndex() {}
    ~KDTreeIndex() {}
    // Implement the methods here
};

class RPTreeIndex : public TreeIndex
{
public:
    static RPTreeIndex &GetInstance();

private:
    RPTreeIndex() {}
    ~RPTreeIndex() {}
    // Implement the methods here
};

#endif // TREEINDEX_H