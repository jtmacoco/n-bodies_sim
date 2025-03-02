#include "particle.h"
std::shared_ptr<float[]> Partcile::getVertex()
{
    std::shared_ptr<float[]> vertex_data(new float[2]);
    vertex_data[0] = x;
    vertex_data[1] = y;
    return vertex_data;
}