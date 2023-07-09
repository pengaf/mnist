set(ProjectName regression)

set(AllFiles 
	"regression.cpp"
)
#message()
add_definitions(-DCMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_executable(${ProjectName} ${AllFiles})
