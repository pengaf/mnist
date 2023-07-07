set(ProjectName mnist2bmp)

set(AllFiles 
	"mnist2bmp.cpp"
)
#message()
add_definitions(-DCMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_executable(${ProjectName} ${AllFiles})
