#include <queue> 

using namespace std; 

#define DEFAULT_CAPACITY        5

typedef vector<double>  polyfit_t;

class PolyfitQueue 
{
public:
    PolyfitQueue() { capacity = DEFAULT_CAPACITY; clear();}
    PolyfitQueue(int cap) { capacity = cap; clear(); }
    ~PolyfitQueue() {}

    void clear() {
        que.empty();
        sum = {0, 0, 0};
    }

    void init(polyfit_t fit) {
        clear();
        que.push(fit);
        for (int i=0; i<fit.size(); i++) {
            sum[i] += fit[i];
        }
    }

    void append(polyfit_t fit) {
        if (que.size() < capacity) {
            que.push(fit);
        }
        else {
            polyfit_t old_fit = que.front();
            que.pop();
            que.push(fit);
            for (int i=0; i<old_fit.size(); i++) {
                sum[i] -= old_fit[i];
            }
        }
        for (int i=0; i<fit.size(); i++) {
            sum[i] += fit[i];
        }
    }

    polyfit_t mean() {
        polyfit_t avg;
        int size = que.size();
        for (int i=0; i<sum.size(); i++) {
            avg.push_back(sum[i]/size);            
        }
        return avg;
    }

private:
    queue<polyfit_t> que;
    int capacity;
    polyfit_t sum;
}; 