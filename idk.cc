//
// Created by Shah Rukh Qasim on 2019-11-04.
//

#include <new>
#include <iostream>
#include <vector>
#include <queue>

using namespace std;

// Redefinition
struct combined {
    int index;
    float distance;
};
class combinedcomparator {
public:
    int operator() (const combined& p1, const combined& p2)
    {
        return p1.distance < p2.distance;
    }
};



int main() {
    std::vector<float> data = {2, 8, 7, 5, 9, 3, 6, 1, 10, 4};
    int k=4;

    std::priority_queue <combined, std::vector<combined>, combinedcomparator> topneighbors;
    for (int i = 0; i < data.size(); i++) {
        if (topneighbors.size()<k) {
            topneighbors.push({i, data[i]});
        }
        else if (topneighbors.top().distance > data[i]) {
            topneighbors.pop();
            topneighbors.push({i, data[i]});
        }
    }

    std::vector<float> res(k);
    for (int i = 0; i < k; ++i) {
        res[k - i - 1] = topneighbors.top().distance;
        topneighbors.pop();
    }
    for (int i = 0; i < k; ++i) {
        std::cout<< res[i] <<std::endl;
    }




    return 0;
}