#pragma once
#include "pagani/graph.hpp"

std::vector<double> dispatch_weighted_bc(const pagani::Csr& graph, const std::string& mode);
