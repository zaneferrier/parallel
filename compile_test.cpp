#include "execution_policy.hpp"
#include "all_any_none.hpp"
#include "equal.hpp"
#include "for_each.hpp"
#include "count.hpp"

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

    bool r = exp_par::any_of(p, v.begin(), v.end(), [](int i) { return i >= 0; });
    std::cout << std::boolalpha << r << '\n';

    r = exp_par::all_of(p, v.begin(), v.end(), [](int i) { return i >= 0; });
    std::cout << std::boolalpha << r << '\n';

    r = exp_par::none_of(p, v.begin(), v.end(), [](int i) { return i < 0; });
    std::cout << std::boolalpha << r << '\n';

    exp_par::for_each(p, v.begin(), v.end(), [](int i) { if((i % 10000) == 0) std::cout << i << '\n'; });
    auto num = exp_par::count(p, v.begin(), v.end(), 5000);
    std::cout << num << '\n';

    num = exp_par::count_if(p, v.begin(), v.end(), [](int i) { return i % 2 == 0; });
    std::cout << num << '\n';
}
