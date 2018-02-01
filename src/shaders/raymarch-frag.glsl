#version 300 es

precision highp float;

uniform vec2 u_AspectRatio;
uniform float u_Time;

in vec4 fs_Pos;
out vec4 out_Col;

const int MAX_MARCHING_STEPS = 255;
const float MIN_DIST = 0.0;
const float MAX_DIST = 100.0;
const float EPSILON = 0.0001;


/**
 * Rotation matrix around the X axis.
 */
mat3 rotateX(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(1, 0, 0),
        vec3(0, c, -s),
        vec3(0, s, c)
    );
}

/**
 * Rotation matrix around the Y axis.
 */
mat3 rotateY(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, 0, s),
        vec3(0, 1, 0),
        vec3(-s, 0, c)
    );
}

/**
 * Rotation matrix around the Z axis.
 */
mat3 rotateZ(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, -s, 0),
        vec3(s, c, 0),
        vec3(0, 0, 1)
    );
}

float intersectSDF(float distA, float distB) {
    return max(distA, distB);
}

float unionSDF(float distA, float distB) {
    return min(distA, distB);
}

float differenceSDF(float distA, float distB) {
    return max(distA, -distB);
}

// polynomial smooth min (k = 0.1);
float smin1( float a, float b, float k ) {
    float h = clamp(0.5+0.5*(b-a)/k, 0.0, 1.0);
    return mix(b, a, h) - k*h*(1.0-h);
}

// exponential smooth min (k = 32);
float smin2( float a, float b, float k )
{
    float res = exp( -k*a ) + exp( -k*b );
    return -log( res )/k;
}

// power smooth min (k = 8);
float smin3( float a, float b, float k )
{
    a = pow( a, k ); b = pow( b, k );
    return pow( (a*b)/(a+b), 1.0/k );
}

float smoothUnionSDF(float distA, float distB) {
    return smin1(distA, distB, 0.3);
}


/**
 * Signed distance function for a sphere centered at the origin with radius 1.0;
 */
float sphereSDF(vec3 p) {
    return length(p) - 1.0;
}

float boxSDF(vec3 p, vec3 b) {
  vec3 d = abs(p) - b;
  return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

float udRoundBox(vec3 p, vec3 b, float r) {
  return length(max(abs(p)-b,0.0))-r;
}

/**
 * Signed distance function for a cube centered at the origin
 * with width = height = length = 2.0
 */
float cubeSDF(vec3 p) {
    // If d.x < 0, then -1 < p.x < 1, and same logic applies to p.y, p.z
    // So if all components of d are negative, then p is inside the unit cube
    vec3 d = abs(p) - vec3(1.0, 1.0, 1.0);
    
    // Assuming p is inside the cube, how far is it from the surface?
    // Result will be negative or zero.
    float insideDistance = min(max(d.x, max(d.y, d.z)), 0.0);
    
    // Assuming p is outside the cube, how far is it from the surface?
    // Result will be positive or zero.
    float outsideDistance = length(max(d, 0.0));
    
    return insideDistance + outsideDistance;
}

/**
 * Signed distance function for an XY aligned cylinder centered at the origin with
 * height h and radius r.
 */
float cylinderSDF(vec3 p, float h, float r) {
    // How far inside or outside the cylinder the point is, radially
    float inOutRadius = length(p.xy) - r;
    
    // How far inside or outside the cylinder is, axially aligned with the cylinder
    float inOutHeight = abs(p.z) - h/2.0;
    
    // Assuming p is inside the cylinder, how far is it from the surface?
    // Result will be negative or zero.
    float insideDistance = min(max(inOutRadius, inOutHeight), 0.0);

    // Assuming p is outside the cylinder, how far is it from the surface?
    // Result will be positive or zero.
    float outsideDistance = length(max(vec2(inOutRadius, inOutHeight), 0.0));
    
    return insideDistance + outsideDistance;
}

float cappedCylinderSDF( vec3 p, vec2 h )
{
  vec2 d = abs(vec2(length(p.xz),p.y)) - h;
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float toastSDF(vec3 p) {
	float toastCylinder1 = cylinderSDF((rotateY(1.5708) * p) + vec3(0.5, -2.0, 0.0), 0.3, 0.6);
	float toastCylinder2 = cylinderSDF((rotateY(1.5708) * p) + vec3(-0.5, -2.0, 0.0), 0.3, 0.6);
	float toastCylinders = unionSDF(toastCylinder1, toastCylinder2);
	float toastBody = boxSDF(p + vec3(0.0, -1.3, 0.0), vec3(0.15, 0.6, 1.0));
	return unionSDF(toastCylinders, toastBody);
}

float repeatToastSDF(vec3 p, vec3 c) {
    vec3 q = mod(p,c)-0.5*c;
    return toastSDF(q);
}

float wheelHoleSDF(vec3 p) {
	return cylinderSDF(rotateY(1.5708) * p, 0.5, 0.6);
}

float toasterBodySDF(vec3 p) {
	float roundBoxDist = udRoundBox(p, vec3(1.0, 1.1, 1.8), 0.2);
	float hole1Dist = udRoundBox(p + vec3(-0.4, -0.2, 0.0), vec3(0.2, 1.1, 1.5), 0.05);
	float hole2Dist = udRoundBox(p + vec3(0.4, -0.2, 0.0), vec3(0.2, 1.1, 1.5), 0.05);
	float holesDist = unionSDF(hole1Dist, hole2Dist);
	float leverHoleDist = boxSDF(p + vec3(0.0, -0.2, -1.8), vec3(0.05, 0.6, 0.3));
	holesDist = unionSDF(holesDist, leverHoleDist);
	float toaster =  differenceSDF(roundBoxDist, holesDist);

	float toasterMinus = boxSDF(p + vec3(0.0, 1.2, 0.0), vec3(1.4, 0.125, 2.1));
	toaster = differenceSDF(toaster, toasterMinus);

	float wheelHole1 = wheelHoleSDF(p + vec3(-1.0, 0.9, 0.9));
	float wheelHole2 = wheelHoleSDF(p + vec3(-1.0, 0.9, -0.9));
	float wheelHole3 = wheelHoleSDF(p + vec3(1.0, 0.9, 0.9));
	float wheelHole4 = wheelHoleSDF(p + vec3(1.0, 0.9, -0.9));
	float wheelHoles = unionSDF(unionSDF(wheelHole1, wheelHole2), unionSDF(wheelHole3, wheelHole4));
	toaster = differenceSDF(toaster, wheelHoles);

	return toaster;
}

float wheelSDF(vec3 p) {
	return cylinderSDF(rotateY(1.5708) * p, 0.5, 0.5);
}

/**
 * Signed distance function describing the scene.
 * 
 * Absolute value of the return value indicates the distance to the surface.
 * Sign indicates whether the point is inside or outside the surface,
 * negative indicating inside.
 */
float sceneSDF(vec3 p) {
	float leverHandle = udRoundBox(p + vec3(0.0, -0.6, -2.15), vec3(0.2, 0.06, 0.05), 0.1);
	
	float toaster = unionSDF(toasterBodySDF(p), leverHandle);

	float dial = cylinderSDF(rotateZ(-0.61) * (p + vec3(-0.65, 0.6, -1.95)), 0.3, 0.2);
	float dialHandle = boxSDF(rotateZ(-0.61) * (p + vec3(-0.65, 0.6, -2.05)), vec3(0.18, 0.02, 0.2));
	dial = smoothUnionSDF(dial, dialHandle);
	toaster = unionSDF(toaster, dial);

	float wheelSpoke1 = cylinderSDF((rotateY(1.5708) * p) + vec3(0.9, 0.9, 0.0), 2.7, 0.1);
	float wheelSpoke2 = cylinderSDF((rotateY(1.5708) * p) + vec3(-0.9, 0.9, 0.0), 2.7, 0.1);
	toaster = unionSDF(toaster, unionSDF(wheelSpoke1, wheelSpoke2));

	float wheel1 = wheelSDF(p + vec3(-1.0, 0.9, 0.9));
	toaster = unionSDF(toaster, wheel1);

	float toast1 = toastSDF(p + vec3(0.4, 0.0, 0.0));
	float toast2 = toastSDF(p + vec3(-0.4, 0.0, 0.0));
	return unionSDF(toaster, unionSDF(toast1, toast2));

}

/**
 * Return the shortest distance from the eyepoint to the scene surface along
 * the marching direction. If no part of the surface is found between start and end,
 * return end.
 * 
 * eye: the eye point, acting as the origin of the ray
 * marchingDirection: the normalized direction to march in
 * start: the starting distance away from the eye
 * end: the max distance away from the ey to march before giving up
 */
float shortestDistanceToSurface(vec3 eye, vec3 marchingDirection, float start, float end) {
    float depth = start;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        float dist = sceneSDF(eye + depth * marchingDirection);
        if (dist < EPSILON) {
			return depth;
        }
        depth += dist;
        if (depth >= end) {
            return end;
        }
    }
    return end;
}
            

/**
 * Return the normalized direction to march in from the eye point for a single pixel.
 * 
 * fieldOfView: vertical field of view in degrees
 * size: resolution of the output image
 * fragCoord: the x,y coordinate of the pixel in the output image
 */
vec3 rayDirection(float fieldOfView, vec2 size, vec2 fragCoord) {
    vec2 xy = fragCoord - size / 2.0;
    float z = size.y / tan(radians(fieldOfView) / 2.0);
    return normalize(vec3(xy, -z));
}

/**
 * Using the gradient of the SDF, estimate the normal on the surface at point p.
 */
vec3 estimateNormal(vec3 p) {
    return normalize(vec3(
        sceneSDF(vec3(p.x + EPSILON, p.y, p.z)) - sceneSDF(vec3(p.x - EPSILON, p.y, p.z)),
        sceneSDF(vec3(p.x, p.y + EPSILON, p.z)) - sceneSDF(vec3(p.x, p.y - EPSILON, p.z)),
        sceneSDF(vec3(p.x, p.y, p.z  + EPSILON)) - sceneSDF(vec3(p.x, p.y, p.z - EPSILON))
    ));
}

/**
 * Lighting contribution of a single point light source via Phong illumination.
 * 
 * The vec3 returned is the RGB color of the light's contribution.
 *
 * k_a: Ambient color
 * k_d: Diffuse color
 * k_s: Specular color
 * alpha: Shininess coefficient
 * p: position of point being lit
 * eye: the position of the camera
 * lightPos: the position of the light
 * lightIntensity: color/intensity of the light
 *
 * See https://en.wikipedia.org/wiki/Phong_reflection_model#Description
 */
vec3 phongContribForLight(vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye,
                          vec3 lightPos, vec3 lightIntensity) {
    vec3 N = estimateNormal(p);
    vec3 L = normalize(lightPos - p);
    vec3 V = normalize(eye - p);
    vec3 R = normalize(reflect(-L, N));
    
    float dotLN = dot(L, N);
    float dotRV = dot(R, V);
    
    if (dotLN < 0.0) {
        // Light not visible from this point on the surface
        return vec3(0.0, 0.0, 0.0);
    } 
    
    if (dotRV < 0.0) {
        // Light reflection in opposite direction as viewer, apply only diffuse
        // component
        return lightIntensity * (k_d * dotLN);
    }
    return lightIntensity * (k_d * dotLN + k_s * pow(dotRV, alpha));
}

/**
 * Lighting via Phong illumination.
 * 
 * The vec3 returned is the RGB color of that point after lighting is applied.
 * k_a: Ambient color
 * k_d: Diffuse color
 * k_s: Specular color
 * alpha: Shininess coefficient
 * p: position of point being lit
 * eye: the position of the camera
 *
 * See https://en.wikipedia.org/wiki/Phong_reflection_model#Description
 */
vec3 phongIllumination(vec3 k_a, vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye) {
    const vec3 ambientLight = 0.5 * vec3(1.0, 1.0, 1.0);
    vec3 color = ambientLight * k_a;
    
    vec3 light1Pos = vec3(4.0 * sin(u_Time * 0.01), 2.0, 4.0 * cos(u_Time * 0.01));
    vec3 light1Intensity = vec3(0.4, 0.4, 0.4);
    
    color += phongContribForLight(k_d, k_s, alpha, p, eye,
                                  light1Pos,
                                  light1Intensity);
    
    vec3 light2Pos = vec3(2.0 * sin(0.37 * u_Time * 0.01),
                          2.0 * cos(0.37 * u_Time * 0.01),
                          2.0);
    vec3 light2Intensity = vec3(0.4, 0.4, 0.4);
    
    color += phongContribForLight(k_d, k_s, alpha, p, eye,
                                  light2Pos,
                                  light2Intensity);    
    return color;
}


/**
 * Return a transform matrix that will transform a ray from view space
 * to world coordinates, given the eye point, the camera target, and an up vector.
 *
 * This assumes that the center of the camera is aligned with the negative z axis in
 * view space when calculating the ray marching direction. See rayDirection.
 */
mat4 viewMatrix(vec3 eye, vec3 center, vec3 up) {
    // Based on gluLookAt man page
    vec3 f = normalize(center - eye);
    vec3 s = normalize(cross(f, up));
    vec3 u = cross(s, f);
    return mat4(
        vec4(s, 0.0),
        vec4(u, 0.0),
        vec4(-f, 0.0),
        vec4(0.0, 0.0, 0.0, 1)
    );
}

void main() {
	vec2 fragCoord = ((fs_Pos.xy + 1.0) / 2.0) * u_AspectRatio.xy;
	
	vec3 viewDir = rayDirection(45.0, u_AspectRatio.xy, fragCoord);
    
	vec3 eye = vec3(15.0, -10.0, 7.0);
    mat4 viewToWorld = viewMatrix(eye, vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));
    vec3 worldDir = (viewToWorld * vec4(viewDir, 0.0)).xyz;
    
    float dist = shortestDistanceToSurface(eye, worldDir, MIN_DIST, MAX_DIST);
    
    if (dist > MAX_DIST - EPSILON) {
        // Didn't hit anything
        out_Col = vec4(0.0, 0.0, 0.0, 1.0);
		return;
    }
    
    // The closest point on the surface to the eyepoint along the view ray
    vec3 p = eye + dist * worldDir;
    
    // Use the surface normal as the ambient color of the material
    vec3 K_a = (estimateNormal(p) + vec3(1.0)) / 2.0;
    vec3 K_d = K_a;
    vec3 K_s = vec3(1.0, 1.0, 1.0);
    float shininess = 10.0;
    
    vec3 color = phongIllumination(K_a, K_d, K_s, shininess, p, eye);
    
    out_Col = vec4(color, 1.0);

}
