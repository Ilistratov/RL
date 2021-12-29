set(Utill_SRC
  logger.cpp
)

add_library(Utill ${Utill_SRC})
target_link_libraries(Utill RLCommon)
