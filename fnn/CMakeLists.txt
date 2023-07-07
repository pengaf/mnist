set(ProjectName fnn)

set(AllFiles 
	"fnn.cpp"
)
#message()
add_definitions(-DCMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_executable(${ProjectName} ${AllFiles})
