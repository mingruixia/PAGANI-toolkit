# pagani_tool(<target> <output name> <unweighted|weighted> <sources...> [DEFS <definitions...>] [LIBS <libs...>])
function(pagani_tool target outname variant)
  cmake_parse_arguments(T "" "" "SOURCES;DEFS;LIBS" ${ARGN})
  add_executable(${target} ${T_SOURCES})
  target_link_libraries(${target} PRIVATE pagani_metrics ${T_LIBS})
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_FUNCTION_LIST_DIR})
  if(variant STREQUAL "weighted")
    target_compile_definitions(${target} PRIVATE PAGANI_WEIGHTED ${T_DEFS})
    set(outdir ${PAGANI_WEIGHTED_DIR})
  else()
    target_compile_definitions(${target} PRIVATE ${T_DEFS})
    set(outdir ${PAGANI_UNWEIGHTED_DIR})
  endif()
  set_target_properties(${target} PROPERTIES OUTPUT_NAME ${outname}
    RUNTIME_OUTPUT_DIRECTORY ${outdir}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${outdir}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${outdir}
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${outdir})
  install(TARGETS ${target} RUNTIME DESTINATION $<IF:$<STREQUAL:${variant},weighted>,exefiles_weighted,exefiles>)
endfunction()
set(PAGANI_APPS_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

foreach(v unweighted weighted)
  pagani_tool(degree_${v}      Degree             ${v} SOURCES degree.cpp)
  pagani_tool(cp_${v}          Cp                 ${v} SOURCES clustering.cpp)
  pagani_tool(pc_${v}          PC_CPU             ${v} SOURCES participation.cpp)
  pagani_tool(louvain_${v}     Louvain_Modularity ${v} SOURCES modularity.cpp DEFS PAGANI_MODULARITY_TOOL="Louvain_Modularity")
  pagani_tool(smallworld_${v}  SmallWorldProperty ${v} SOURCES smallworld.cpp)
  pagani_tool(convertnii_${v}  ConvertNII         ${v} SOURCES convertnii.cpp)
  pagani_tool(maslov_${v}      Maslov             ${v} SOURCES maslov.cpp)
  pagani_tool(bfslp_${v}       BFS_MulCPU         ${v} SOURCES lp_cpu.cpp)
endforeach()
pagani_tool(newman_cpu_unweighted Newman_Modularity_CPU unweighted SOURCES modularity.cpp
            DEFS PAGANI_NEWMAN PAGANI_MODULARITY_TOOL="Newman_Modularity_CPU")
pagani_tool(newman_cpu_weighted   Newman_Modularity_CPU weighted   SOURCES modularity.cpp
            DEFS PAGANI_NEWMAN PAGANI_MODULARITY_TOOL="Newman_Modularity_CPU")
