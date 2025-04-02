CXX = g++
CPPFLAGS = -g
CPPWARNINGS = -Wall -pedantic
DEPDIR = deps
OBJDIR = objs
LIBS = -lGLEW -lGL -lGLU -lglut -lglfw  
OMPFLAGS = -fopenmp
# Find all .cpp files and their respective directories
CPPSOURCEFILES = $(shell find . -name '*.cpp')
# Get the object files paths corresponding to source files
OBJFILES = $(CPPSOURCEFILES:%.cpp=$(OBJDIR)/%.o)
DEPFILES = $(CPPSOURCEFILES:%.cpp=$(DEPDIR)/%.d)
main: $(OBJFILES)
	$(CXX) $(OMPFLAGS) -g -o main $(OBJFILES)  $(LIBS)

$(OBJDIR)/%.o : %.cpp $(DEPDIR)/%.d | $(DEPDIR) $(OBJDIR)
	$(CXX) $(OMPFLAGS) $(DEPFLAGS) $(CPPFLAGS) $(CPPWARNINGS) $(MACRO_DEFINITIONS) -o $@ -c $<
# Create the necessary directories
$(DEPDIR): ; @mkdir -p $@
$(OBJDIR): ; @mkdir -p $@


# Include dependency files
$(DEPFILES):
include $(wildcard $(DEPFILES))

# Clean up generated files
clean:
	rm -rf $(OBJDIR) $(DEPDIR) main