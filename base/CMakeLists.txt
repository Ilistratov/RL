set(Base_SRC
  base.cpp
  context.cpp
)

add_library(Base ${Base_SRC})
target_link_libraries(Base RLCommon)
