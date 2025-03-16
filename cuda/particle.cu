#include "particle.h"
float3 Particle::getPosition()
{
    return pos;
}
float Particle::getMass(){
    return mass;
}
void Particle::setVelocity(float3 v){
    velocity.x -= v.x;
    velocity.y -= v.y;
    velocity.z -= v.z;
}
float3 Particle::getAcceleration(){
    return acceleration;
}
float3 Particle::getVelocity(){
    return velocity;
}
void Particle::setPosition(float3 p){
    pos.x -= p.x;
    pos.y -= p.y;
    pos.z -= p.z;
}
float Particle::getForce(){
    return force;
}
void Particle::applyForce(float3 incoming_force, float dt){

    pos.x += (velocity.x * dt);
    pos.y += (velocity.y * dt);
    pos.z += (velocity.z * dt);


    velocity.x += (acceleration.x * dt);
    velocity.y += (acceleration.y * dt);
    velocity.z += (acceleration.z * dt);

    incoming_force.x/=mass;
    incoming_force.y/=mass;
    incoming_force.z/=mass;

    acceleration = incoming_force;
}