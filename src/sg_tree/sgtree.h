/*
 * Copyright (c) 2017 Manzil Zaheer All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Copyright 2020 Google LLC.
 * SPDX-License-Identifier: Apache-2.0
 */


# ifndef _COVER_TREE_H
# define _COVER_TREE_H

//#define DEBUG

#include <atomic>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <shared_mutex>
#include <stack>
#include <vector>

#include <Eigen/Core>

#include "utils.h"

class CoverTree
{
/************************* Internal Functions ***********************************************/
protected:
    /*** Base to use for the calculations ***/
    static constexpr scalar base = 1.3;
    static scalar* compute_pow_table();
    static scalar* powdict;

public:
    /*** structure for each node ***/
    struct Node
    {
        pointType _p;                       // point associated with the node
        std::vector<Node*> children;        // list of children
        int level;                          // current level of the node
        double maxdistUB;                   // upper bound of distance to any of descendants
        unsigned ID;                        // unique ID of current node
        unsigned UID;                       // external unique ID for current node

        mutable std::shared_mutex mut;      // lock for current node

        /*** Node modifiers ***/
        double covdist()                    // covering distance of subtree at current node
        {
            return powdict[level + 1024];
        }
        double sepdist()                    // separating distance between nodes at current level
        {
            return powdict[level + 1023];
        }
        double dist(const pointType& pp) const  // L2 distance between current node and point pp
        {
            return (_p - pp).norm();
        }
        double dist(Node* n) const              // L2 distance between current node and node n
        {
            return (_p - n->_p).norm();
        }
        Node* setChild(const pointType& pIns, 
                       unsigned UID = 0,
                       int new_id = -1)     // insert a new child of current node with point pIns
        {
            Node* temp = new Node;
            temp->_p = pIns;
            temp->level = level - 1;
            temp->maxdistUB = 0;
            temp->ID = new_id;
            temp->UID = UID;
            children.push_back(temp);
            return temp;
        }

        /*** erase child ***/
        void erase(size_t pos)
        {
            children[pos] = children.back();
            children.pop_back();
        }

        void erase(std::vector<Node*>::iterator pos)
        {
            *pos = children.back();
            children.pop_back();
        }

        /*** Iterator access ***/
        inline std::vector<Node*>::iterator begin()
        {
            return children.begin();
        }
        inline std::vector<Node*>::iterator end()
        {
            return children.end();
        }
        inline std::vector<Node*>::const_iterator begin() const
        {
            return children.begin();
        }
        inline std::vector<Node*>::const_iterator end() const
        {
            return children.end();
        }

        /*** Pretty print ***/
        friend std::ostream& operator<<(std::ostream& os, const Node& ct)
        {
            if (ct._p.rows() < 6)
            {
                Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");
                os << "(" << ct._p.format(CommaInitFmt) << ":" << ct.level << ":" << ct.maxdistUB <<  ":" << ct.ID << ")";
            }
            else
            {
                Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
                os << "([" << ct._p.head<3>().format(CommaInitFmt) << ", ..., " << ct._p.tail<3>().format(CommaInitFmt) << "]:" << ct.level << ":" << ct.maxdistUB <<  ":" << ct.ID << ")";
            }
            return os;
        }
    };

protected:
    Node* root;                         // Root of the tree
    std::atomic<int> min_scale;         // Minimum scale
    std::atomic<int> max_scale;         // Minimum scale
    int truncate_level;                 // Relative level below which the tree is truncated
    bool id_valid;

    std::atomic<unsigned> N;            // Number of points in the cover tree
    unsigned D;                         // Dimension of the points

    std::shared_mutex global_mut;       // lock for changing the root

    /*** Insertion helper function ***/
    bool insert(Node* current, const pointType& p, unsigned UID);

    /*** Serialize/Desrialize helper function ***/
    char* preorder_pack(char* buff, Node* current) const;       // Pre-order traversal
    char* postorder_pack(char* buff, Node* current) const;      // Post-order traversal
    void PrePost(Node*& current, char*& pre, char*& post);

public:
    /*** Internal Contructors ***/
    /*** Constructor: needs at least 1 point to make a valid cover-tree ***/
    // NULL tree
    explicit CoverTree(int truncate = -1);
    // cover tree with one point as root
    CoverTree(const pointType& p, int truncate = -1);
    // cover tree using points in the list between begin and end
    CoverTree(std::vector<pointType>& pList, int begin, int end, int truncate = -1, bool use_multi_core = true);
    // cover tree using points in the list between begin and end
    CoverTree(matrixType& pMatrix, int begin, int end, int truncate = -1, bool use_multi_core = true);
    // cover tree using points in the list between begin and end
    CoverTree(Eigen::Map<matrixType>& pMatrix, int begin, int end, int truncate = -1, bool use_multi_core = true);

    /*** Destructor ***/
    /*** Destructor: deallocating all memories by a post order traversal ***/
    ~CoverTree();

/************************* Public API ***********************************************/
public:
    /*** construct cover tree using all points in the list ***/
    static CoverTree* from_points(std::vector<pointType>& pList, int truncate = -1, bool use_multi_core = true);

    /*** construct cover tree using all points in the matrix in row-major form ***/
    static CoverTree* from_matrix(matrixType& pMatrix, int truncate = -1, bool use_multi_core = true);

    /*** construct cover tree using all points in the matrix in row-major form ***/
    static CoverTree* from_matrix(Eigen::Map<matrixType>& pMatrix, int truncate = -1, bool use_multi_core = true);


    /*** Insert point p into the cover tree ***/
    bool insert(const pointType& p, unsigned UID = 0);

    /*** Remove point p from the cover tree ***/
    bool remove(const pointType& p);

    /*** Nearest Neighbour search ***/
    std::pair<CoverTree::Node*, double> NearestNeighbour(const pointType &p) const;

    /*** k-Nearest Neighbour search ***/
    std::vector<std::pair<CoverTree::Node*, double>> kNearestNeighbours(const pointType &p, unsigned k = 10) const;

    /*** Range search ***/
    std::vector<std::pair<CoverTree::Node*, double>> rangeNeighbours(const pointType &p, double range = 1.0) const;

    /*** Serialize/Desrialize: useful for MPI ***/
    char* serialize() const;                                    // Serialize to a buffer
    size_t msg_size() const;
    void deserialize(char* buff);                               // Deserialize from a buffer

    /*** Unit Tests ***/
    bool check_covering() const;
    void print_stats() const;
    void print_levels() const;
    void print_degrees() const;

    /*** Pretty print ***/
    friend std::ostream& operator<<(std::ostream& os, const CoverTree& ct);
};

#endif //_COVER_TREE_H

