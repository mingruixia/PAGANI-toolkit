// SmallWorldProperty.exe dir
// sigma = gamma / lambda for every network that has both <network>_gamma.txt
// (from Cp.exe ... k) and <network>_lambda.txt (from CUBFW_Lp.exe ... l).
// Output: <network>_smallWorld.txt
//
// Fix (audit S-01): files are paired by their full network name instead of a
// few characters of the file name, and any number of files is supported.
#include <fstream>

#include "app_util.hpp"

using namespace pagani;

static double read_first_value(const std::string& path) {
    std::ifstream f(path);
    double v;
    if (!(f >> v)) throw Error("cannot read a number from " + path);
    return v;
}

int main(int argc, char** argv) {
    return run_tool("SmallWorldProperty", argc, argv, [](int argc, char** argv) {
        if (argc != 2) throw UsageError("SmallWorldProperty.exe dir\n  (run Lp with flag l and Cp with flag k first)");
        const std::string dir = argv[1];
        auto lambdas = list_files_with_suffix(dir, "_lambda.txt");
        auto gammas = list_files_with_suffix(dir, "_gamma.txt");
        std::cout << lambdas.size() << " lambda file(s), " << gammas.size() << " gamma file(s)" << std::endl;
        std::set<std::string> gamma_stems;
        for (const auto& gname : gammas) gamma_stems.insert(gname.substr(0, gname.size() - 10));
        int done = 0;
        std::set<std::string> used;
        for (const auto& l : lambdas) {
            const std::string stem = l.substr(0, l.size() - 11);
            if (!gamma_stems.count(stem)) {
                std::cout << "no gamma file for " << stem << " (skipped)" << std::endl;
                continue;
            }
            used.insert(stem);
            const double lambda = read_first_value(path_join(dir, l));
            const double gamma = read_first_value(path_join(dir, stem + "_gamma.txt"));
            const double sigma = gamma / lambda;
            std::ofstream o(path_join(dir, stem + "_smallWorld.txt"));
            o << fmt6(sigma) << "\n";
            if (!o) throw Error("cannot write " + stem + "_smallWorld.txt");
            std::cout << stem << ":  gamma = " << fmt6(gamma) << "  lambda = " << fmt6(lambda)
                      << "  sigma = " << fmt6(sigma) << std::endl;
            ++done;
        }
        for (const auto& s : gamma_stems)
            if (!used.count(s)) std::cout << "no lambda file for " << s << " (skipped)" << std::endl;
        if (done == 0) throw Error("no matching lambda/gamma pairs found in " + dir);
        return 0;
    });
}
