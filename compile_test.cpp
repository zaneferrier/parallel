#include "execution_policy.hpp"
//#include "all_of.hpp"
#include "equal.hpp"

#include <iostream>

// Just a test file so I can check that everything at least compiles,
// and the most basic of basic tests give the correct results.

namespace exp_par = experimental::parallel;

int main()
{
    exp_par::execution_policy p = exp_par::seq;
     
    std::vector<int> v;
    std::vector<int> t;
    
    for(auto i = 0; i < 100000; ++i) {
        v.push_back(i);
        t.push_back(i);
    }

    p = exp_par::par;

    bool result = exp_par::equal(p, v.begin(), v.end(), t.begin(), t.end());        
    std::cout << std::boolalpha << result << '\n';

/*
    bool r = parallel::any_of(p, v.begin(), v.end(), [](int i) { return i >= 0; });
    std::cout << std::boolalpha << r << '\n';

    r = parallel::all_of(p, v.begin(), v.end(), [](int i) { return i >= 0; });
    std::cout << std::boolalpha << r << '\n';
*/
}
