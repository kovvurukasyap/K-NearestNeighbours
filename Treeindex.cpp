#include <bits/stdc++.h>
#include "VectorDataset.h"
#include "DataVector.h"
#include "DataVector.cpp"
using namespace std;

int THRESHOLD = 1000;

class TreeIndex
{
protected:
    vector<DataVector> all_vectors;

    TreeIndex(vector<DataVector>::iterator start, vector<DataVector>::iterator end) : all_vectors(start, end) {}

    TreeIndex()
    {
        VectorDataset temp;
        temp.ReadDataset("fmnist-train.csv");
        all_vectors.clear();
        int l = temp.v.size();
        for (int i = 0; i < l; i++)
        {
            all_vectors.push_back(temp.v[i]);
        }
        cout << "TreeIndex Constructor called\n";
    }

    ~TreeIndex() { cout << "TreeIndex Destructor called\n"; }

public:
    static TreeIndex &GetInstance()
    {
        static TreeIndex ins;
        return ins;
    }

    void ReadData(vector<DataVector> &arr)
    {
        all_vectors.clear();
        for (int i = 0; i < arr.size(); i++)
        {
            all_vectors.push_back(arr[i]);
        }
    }

    vector<DataVector> &GetData()
    {
        return all_vectors;
    }
};

class KDTreeIndex : public TreeIndex
{
public:
    struct Node
    {
        vector<DataVector> break_info;
        Node *left;
        Node *right;

        Node(vector<DataVector>::iterator begin, vector<DataVector>::iterator end) : break_info(begin, end), left(NULL), right(NULL) {}
    };

    KDTreeIndex()
    {
        TreeIndex::GetInstance();
        cout << "KDTreeIndex Constructor called\n";
    }

    ~KDTreeIndex() { cout << "KDTreeIndex Destructor called\n"; }

public:
    Node *root = NULL;

    static KDTreeIndex &GetInstance()
    {
        static KDTreeIndex ins;
        return ins;
    }

    void Choose(const vector<DataVector> &data, vector<DataVector>::iterator begin, vector<DataVector>::iterator end, int &break_dimn, double &median)
    {
        int dimension = data[0].v.size();
        double maxSpread = -1;

        for (int i = 0; i < dimension; ++i)
        {
            vector<double> values;
            for_each(begin, end, [&values, i](const DataVector &vec)
                     { values.push_back(vec.v[i]); });

            sort(values.begin(), values.end());
            double spread = values[values.size() - 1] - values[0];

            if (spread > maxSpread)
            {
                maxSpread = spread;
                break_dimn = i;
                median = values[(values.size() - 1) / 2];
            }
        }
    }

    Node *KDTree(vector<DataVector> &dataset, vector<DataVector>::iterator begin, vector<DataVector>::iterator end)
    {
        if ((end - begin) <= THRESHOLD)
        {
            return new Node(begin, end);
        }

        int split;
        double median;
        Choose(dataset, begin, end, split, median);
        int left_count = 0, right_count = 0;
        for_each(begin, end, [&split, &median, &left_count, &right_count](const DataVector &vec)
                 {
            if (vec.v[split] <= median) {
                left_count++;
            } });
        sort(begin, end, [split](const DataVector &a, const DataVector &b)
             { return a.v[split] < b.v[split]; });
        vector<DataVector> break_node(2);
        break_node[0].setDimension(2);
        break_node[0].v[0] = split;
        break_node[0].v[1] = median;
        Node *currentNode = new Node(break_node.begin(), break_node.end());

        currentNode->left = KDTree(dataset, begin, begin + left_count);
        currentNode->right = KDTree(dataset, begin + left_count, end);

        return currentNode;
    }

    void ClearTree(Node *node)
    {
        if (node == NULL)
            return;

        ClearTree(node->left);
        ClearTree(node->right);
        delete node;
    }

    void MakeTree()
    {
        ClearTree(root);
        root = KDTree(all_vectors, all_vectors.begin(), all_vectors.end());
    }

    void leaf_node_search(Node *leafNode, const DataVector &test_vector, int k, set<double> &my_set)
    {
        if (leafNode == NULL)
            return;

        for (DataVector &point : leafNode->break_info)
        {
            my_set.insert(point.dist(test_vector));

            if (my_set.size() > k)
            {
                my_set.erase(*my_set.rbegin());
            }
        }
    }

    vector<double> k_nearest(DataVector &test_vector, int k)
    {
        set<double> my_set;

        if (root == NULL)
            return vector<double>(0);

        Node *curr = root;

        while (curr->left != NULL || curr->right != NULL)
        {
            if (test_vector.v[curr->break_info[0].v[0]] > curr->break_info[0].v[1])
            {
                if (curr->right != NULL)
                    curr = curr->right;
                else
                    break;
            }
            else
            {
                if (curr->left != NULL)
                    curr = curr->left;
                else
                    break;
            }
        }
        leaf_node_search(curr, test_vector, k, my_set);
        vector<double> result;
        while (!my_set.empty())
        {
            result.push_back(*my_set.begin());
            my_set.erase(my_set.begin());
        }
        return result;
    }
};

class RPTreeIndex : public TreeIndex
{
public:
    struct Node
    {
        vector<DataVector> break_info;
        Node *left;
        Node *right;

        Node(vector<DataVector>::iterator start, vector<DataVector>::iterator end) : break_info(start, end), left(NULL), right(NULL) {}
    };

    RPTreeIndex()
    {
        TreeIndex::GetInstance();
    }

    ~RPTreeIndex() {}

public:
    Node *root = NULL;
    static RPTreeIndex &GetInstance()
    {
        static RPTreeIndex temp;
        return temp;
    }
    void ClearTree(Node *node)
    {
        if (node == NULL)
            return;

        ClearTree(node->left);
        ClearTree(node->right);
        delete node;
    }
    double max_distance(DataVector &point, const vector<DataVector> &dataset, vector<DataVector>::iterator begin, vector<DataVector>::iterator end)
    {
        double maxDistance = 0.0;
        DataVector farthestPoint;
        for (auto it = begin; it != end; ++it)
        {
            double distance = (point - *it).norm();
            if (distance > maxDistance)
            {
                maxDistance = distance;
                farthestPoint = *it;
            }
        }
        return maxDistance;
    }
    void ChooseRule(vector<DataVector> &dataset, vector<DataVector>::iterator begin, vector<DataVector>::iterator end, double &delta)
    {
        int size = (end - begin);
        DataVector x = *(begin + rand() % size);
        delta = max_distance(x, dataset, begin, end);
        delta *= 6;
        delta /= sqrt(dataset[0].v.size());
        delta *= (((rand() % 200) / 100.0) - 1);
    }
    Node *RPTree(vector<DataVector> &dataset, vector<DataVector>::iterator begin, vector<DataVector>::iterator end)
    {
        if ((end - begin) <= THRESHOLD)
        {
            return new Node(begin, end);
        }
        DataVector _vector(dataset[0].v.size());
        for (int i = 0; i < _vector.v.size(); i++)
        {
            _vector.v[i] = rand();
        }
        double len = _vector.norm();
        for (int i = 0; i < _vector.v.size(); i++)
        {
            _vector.v[i] /= len;
        }
        double delta;
        ChooseRule(dataset, begin, end, delta);
        auto comparator = [&_vector](DataVector &a, DataVector &b)
        {
            return (a * _vector) < (b * _vector);
        };
        sort(begin, end, comparator);
        double median = (*(begin + (end - begin) / 2)) * _vector;
        int left_count = 0;
        for (auto it = begin; it != end; ++it)
        {
            if ((*it) * _vector <= median + delta)
            {
                left_count++;
            }
        }
        vector<DataVector> temp(2);
        temp[0] = _vector;
        temp[1].setDimension(1);
        temp[1].v[0] = median + delta;

        Node *currentNode = new Node(temp.begin(), temp.end());

        currentNode->left = RPTree(dataset, begin, begin + left_count);
        currentNode->right = RPTree(dataset, begin + left_count, end);

        return currentNode;
    }
    void Maketree()
    {
        ClearTree(root);
        root = RPTree(all_vectors, all_vectors.begin(), all_vectors.end());
    }
    void leaf_k_nearest(Node *leafNode, const DataVector &target, int k, set<double> &my_set)
    {
        if (leafNode == nullptr)
            return;

        for (DataVector &point : leafNode->break_info)
        {
            my_set.insert(point.dist(target));

            if (my_set.size() > k)
            {
                my_set.erase(*my_set.rbegin());
            }
        }
    }
    vector<double> k_nearest(DataVector &target, int k)
    {
        set<double> my_set;

        if (root == nullptr)
            return vector<double>(0);
        Node *curr = root;
        while (curr->left != nullptr || curr->right != nullptr)
        {
            if ((target * curr->break_info[0]) > curr->break_info[1].v[0])
            {
                if (curr->right != nullptr)
                    curr = curr->right;
                else
                    break;
            }
            else
            {
                if (curr->left != nullptr)
                    curr = curr->left;
                else
                    break;
            }
        }
        leaf_k_nearest(curr, target, k, my_set);
        vector<double> result;
        while (!my_set.empty())
        {
            result.push_back(*my_set.begin());
            my_set.erase(*my_set.begin());
        }

        return result;
    }
};

int main()
{
    VectorDataset test_vectors;
    test_vectors.ReadDataset("fmnist-test.csv");
    KDTreeIndex KDtreee;
    RPTreeIndex RPtreee;
    KDtreee.MakeTree();
    RPtreee.Maketree();
    auto start_time = chrono::high_resolution_clock::now();
    vector<double> ans = KDtreee.k_nearest(test_vectors.v[99], 10);
    cout << "KDTree--------------------------\n";
    for (int i = 0; i < ans.size(); i++)
    {
        cout << ans[i] << '\n';
    }
    vector<double> ans1 = RPtreee.k_nearest(test_vectors.v[100], 10);
    cout << "RPTree---------------------------\n";
    for (int i = 0; i < ans1.size(); i++)
    {
        cout << ans1[i] << '\n';
    }
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
    cout << "Time taken: " << duration.count() / 1000 << " milliseconds" << endl;
    return 0;
}
