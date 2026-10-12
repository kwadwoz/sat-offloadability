// Exercise the library boundary and its default API phase, including cleanup.
#include "minisat/core/Solver.h"
#include <cassert>
using namespace Minisat;
class Probe : public Solver {
public:
    CRef emptyCall() { return propagate(); }
};
int main(int argc, char** argv) {
    assert(argc == 2);
    Probe s;
    s.openBcpLog(argv[1]);
    Var a = s.newVar(), b = s.newVar();
    s.addClause(~mkLit(a), mkLit(b));
    vec<Lit> assumptions, implied;
    assumptions.push(mkLit(a));
    assert(s.implies(assumptions, implied));
    assert(implied.size() == 1 && implied[0] == mkLit(b));
    assert(s.emptyCall() == CRef_Undef);
    // Destructor must flush/close logs for library users.
}
