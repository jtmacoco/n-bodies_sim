#include "particle.h"
glm::vec3 Particle::getPosition()
{
    return pos;
}
float Particle::getMass(){
    return mass;
}
float Particle::getAcceleration(){
    return acceleration;
}
void Particle::update()
{
    pos.y += 0.001f;
    if(pos.y > 1.0f){
        pos.y = -1.0f;
    }
}