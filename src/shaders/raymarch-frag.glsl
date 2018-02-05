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

const vec3 eye = vec3(12.0, 8.0, 10.0);
const vec3 light1Pos = vec3(6.0, 7.0, 6.0);
const vec3 light2Pos = vec3(-2.0, 7.0, -2.0);



struct Intersection {
	float t;
	vec3 normal;
	vec3 color;
	int id;
};

mat3 rotateX(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(1, 0, 0),
        vec3(0, c, -s),
        vec3(0, s, c)
    );
}

mat3 rotateY(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, 0, s),
        vec3(0, 1, 0),
        vec3(-s, 0, c)
    );
}

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

float smoothUnionSDF(float distA, float distB) {
    return smin1(distA, distB, 0.3);
}

/**
 * Signed distance function for a sphere centered at the origin with radius r
 */
float sphereSDF(vec3 p, float r) {
	return length(p) - r;
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

float ellipsoidSDF(vec3 p, vec3 r) {
    return (length( p/r ) - 1.0) * min(min(r.x,r.y),r.z);
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

	float PI = 3.14159;
	float x = fract(u_Time) * (2.0 * PI);
	float leverHeight = 0.5 * (0.5 -((2.0*floor(x/(2.0*PI)) - floor(2.0*(x/(2.0*PI))) + 1.0)*(sin(-(x-(PI/2.0)))+1.0) + 
		2.0*fract(x/(PI / 2.0)) * (2.0*floor((x-PI)/(2.0*PI)) - floor(2.0*((x-PI)/(2.0*PI))) + 1.0) * 
		(2.0*floor((x+(PI/2.0))/(PI)) - floor(2.0*((x+(PI/2.0))/(PI))) + 1.0))); 
	float leverHandle = udRoundBox(p + vec3(0.0, leverHeight, -2.15), vec3(0.2, 0.06, 0.05), 0.1);
	
	toaster = unionSDF(toaster, leverHandle);

    float dialAngle = -floor(u_Time);
	float dial = cylinderSDF(rotateZ(dialAngle) * (p + vec3(-0.65, 0.6, -1.95)), 0.3, 0.2);
	float dialHandle = boxSDF(rotateZ(dialAngle) * (p + vec3(-0.65, 0.6, -2.05)), vec3(0.18, 0.02, 0.2));
	dial = smoothUnionSDF(dial, dialHandle);

	toaster = unionSDF(toaster, dial);

	return toaster;
}

float wheelSDF(vec3 p) {
    float wheel = cylinderSDF(rotateY(1.5708) * p, 0.5, 0.5);
    wheel = differenceSDF(wheel, cylinderSDF(rotateY(1.5708) * p, 0.6, 0.4));
    float spoke1 = boxSDF(p, vec3(0.25, 0.05, 0.5));
    float spoke2 = boxSDF(rotateX(1.0472) * p, vec3(0.25, 0.05, 0.5));
    float spoke3 = boxSDF(rotateX(2.094) * p, vec3(0.25, 0.05, 0.5));
	return unionSDF(wheel, smoothUnionSDF(spoke1, smoothUnionSDF(spoke2, spoke3)));
}

vec3 mod3(vec3 p, vec3 c) {
	return vec3(mod(p.x, c.x), mod(p.y, c.y), mod(p.z, c.z));
}

float planeSDF(vec3 p, vec4 n){
  return dot(p,n.xyz) + n.w;
}

float sinc( float x, float k ) {
    float a = 3.14159 * (float(k)*x-1.0);
    return sin(a)/a;
}

float outletSDF(vec3 p) {
	float outletCover = udRoundBox(p + vec3(0.0, 0.0, 5.0), vec3(0.5, 0.9, 0.01), 0.05);

	float outletSphereSubtract1 = sphereSDF(p + vec3(0.0, 0.4, 4.95), 0.3);
	float outletBoxSubtract1 = boxSDF(p + vec3(0.0, 0.4, 4.95), vec3(0.5, 0.25, 0.1));
	float outletThingSubtract1 = intersectSDF(outletSphereSubtract1, outletBoxSubtract1);
	float outletSphereSubtract2 = sphereSDF(p + vec3(0.0, -0.4, 4.95), 0.3);
	float outletBoxSubtract2 = boxSDF(p + vec3(0.0, -0.4, 4.95), vec3(0.5, 0.25, 0.1));
	float outletThingSubtract2 = intersectSDF(outletSphereSubtract2, outletBoxSubtract2);
	float outletThingSubtract = unionSDF(outletThingSubtract1, outletThingSubtract2);

	float outlet = differenceSDF(outletCover, outletThingSubtract);

	float outletSphereUnion1 = sphereSDF(p + vec3(0.0, 0.4, 4.95), 0.28);
	float outletBoxUnion1 = boxSDF(p + vec3(0.0, 0.4, 4.95), vec3(0.5, 0.23, 0.05));
	float outletThingUnion1  = intersectSDF(outletSphereUnion1, outletBoxUnion1);
	float outletHole1_1 = boxSDF(p + vec3(0.1, 0.4, 4.9), vec3(0.03, 0.1, 0.05));
	float outletHole1_2 = boxSDF(p + vec3(-0.1, 0.4, 4.9), vec3(0.03, 0.1, 0.05));
	float outletHoles1 = unionSDF(outletHole1_1, outletHole1_2);
	outletThingUnion1 = differenceSDF(outletThingUnion1, outletHoles1);

	float outletSphereUnion2 = sphereSDF(p + vec3(0.0, -0.4, 4.95), 0.28);
	float outletBoxUnion2 = boxSDF(p + vec3(0.0, -0.4, 4.95), vec3(0.5, 0.23, 0.05));
	float outletThingUnion2  = intersectSDF(outletSphereUnion2, outletBoxUnion2);
	float outletHole2_1 = boxSDF(p + vec3(0.1, -0.4, 4.9), vec3(0.03, 0.1, 0.05));
	float outletHole2_2 = boxSDF(p + vec3(-0.1, -0.4, 4.9), vec3(0.03, 0.1, 0.05));
	float outletHoles2 = unionSDF(outletHole2_1, outletHole2_2);
	outletThingUnion2 = differenceSDF(outletThingUnion2, outletHoles2);
	
	float outletThingUnion = unionSDF(outletThingUnion1, outletThingUnion2);

	outlet = unionSDF(outlet, outletThingUnion);

	return outlet;
}

float repeatOutletSDF(vec3 p, vec3 c) {
	vec3 q = mod3(p, c) - 0.5 * c;
    return outletSDF(q);
}

float animatedToastsSDF(vec3 p) {
	float PI = 3.14159;
	float x = fract(u_Time) * (2.0 * PI);    
	float toastHeight = 1.0 + -((2.0*floor(x/(2.0*PI)) - floor(2.0*(x/(2.0*PI))) + 1.0)*(sin(-(x-(PI/2.0)))+1.0) + 
		2.0*fract(x/(PI / 2.0)) * (2.0*floor((x-PI)/(2.0*PI)) - floor(2.0*((x-PI)/(2.0*PI))) + 1.0) * 
		(2.0*floor((x+(PI/2.0))/(PI)) - floor(2.0*((x+(PI/2.0))/(PI))) + 1.0));
	float toast1 = toastSDF(p + vec3(0.4, toastHeight, 0.0));
	float toast2 = toastSDF(p + vec3(-0.4, toastHeight, 0.0));

	return unionSDF(toast1, toast2);
}

float sceneSDF(vec3 p) {
	float toasterBodyJitter = 0.03 * sin(u_Time * 80.0);
	float toaster = toasterBodySDF(p + vec3(0.0, toasterBodyJitter, 0.0));

    float plugHead = ellipsoidSDF(p + vec3(-0.75, 0.4, 4.9), vec3(0.35, 0.3, 0.5));
	float plugHeadSubtract = boxSDF(p + vec3(-0.75, 0.4, 5.3), vec3(0.5, 0.5, 0.4));
	plugHead = differenceSDF(plugHead, plugHeadSubtract);
	float plugHeadIntersect = boxSDF(p + vec3(-0.75, 0.4, 4.6), vec3(0.5, 0.25, 0.9));
	plugHead = intersectSDF(plugHead, plugHeadIntersect);
	float plugCord = cylinderSDF(p + vec3(-0.75, 0.4, 3.5), 3.0, 0.08);
	float plug = smoothUnionSDF(plugHead, plugCord);
	toaster = unionSDF(toaster, plug);

	float wheelAxel1 = cylinderSDF((rotateY(1.5708) * p) + vec3(0.9, 0.9, 0.0), 2.7, 0.1);
	float wheelAxel2 = cylinderSDF((rotateY(1.5708) * p) + vec3(-0.9, 0.9, 0.0), 2.7, 0.1);
	toaster = unionSDF(toaster, unionSDF(wheelAxel1, wheelAxel2));

	mat3 rotateWheels = rotateX(5.0 * u_Time);
	float wheel1 = wheelSDF(rotateWheels * (p + vec3(1.0, 0.9, 0.9)));
	float wheel2 = wheelSDF(rotateWheels * (p + vec3(-1.0, 0.9, -0.9)));
	float wheel3 = wheelSDF(rotateWheels * (p + vec3(-1.0, 0.9, 0.9)));
	float wheel4 = wheelSDF(rotateWheels * (p + vec3(1.0, 0.9, -0.9)));

	float wheels = unionSDF(unionSDF(wheel1, wheel2), unionSDF(wheel3, wheel4));
	toaster = unionSDF(toaster, wheels);

	float toast = animatedToastsSDF(p);

	float outlets = repeatOutletSDF(p, vec3(1.5, 0.0, 0.0));
	return unionSDF(toaster, outlets);
}

vec3 estimateSceneNormal(vec3 p) {
    return normalize(vec3(
        sceneSDF(vec3(p.x + EPSILON, p.y, p.z)) - sceneSDF(vec3(p.x - EPSILON, p.y, p.z)),
        sceneSDF(vec3(p.x, p.y + EPSILON, p.z)) - sceneSDF(vec3(p.x, p.y - EPSILON, p.z)),
        sceneSDF(vec3(p.x, p.y, p.z  + EPSILON)) - sceneSDF(vec3(p.x, p.y, p.z - EPSILON))
    ));
}

vec3 estimateToastsNormal(vec3 p) {
    return normalize(vec3(
        animatedToastsSDF(vec3(p.x + EPSILON, p.y, p.z)) - animatedToastsSDF(vec3(p.x - EPSILON, p.y, p.z)),
        animatedToastsSDF(vec3(p.x, p.y + EPSILON, p.z)) - animatedToastsSDF(vec3(p.x, p.y - EPSILON, p.z)),
        animatedToastsSDF(vec3(p.x, p.y, p.z  + EPSILON)) - animatedToastsSDF(vec3(p.x, p.y, p.z - EPSILON))
    ));
}

vec3 phongContribForLight(vec3 k_d, vec3 k_s, float alpha, vec3 p,
                          vec3 lightPos, vec3 lightIntensity) {
    vec3 N = estimateSceneNormal(p);
    vec3 L = normalize(lightPos - p);
    vec3 V = normalize(eye - p);
    vec3 R = normalize(reflect(-L, N));
    
    float dotLN = dot(L, N);
    float dotRV = dot(R, V);
    
    if (dotLN < 0.0) {
        return vec3(0.0, 0.0, 0.0);
    } 
    
    if (dotRV < 0.0) {
        return lightIntensity * (k_d * dotLN); // it's just a lambert
    }
    return lightIntensity * (k_d * dotLN + k_s * pow(dotRV, alpha));
}


vec3 toasterPhong(vec3 p) {
    vec3 k_a = vec3(0.1, 0.1, 0.1);
    vec3 k_d = vec3(0.98, 0.98, 0.98);
    vec3 k_s = vec3(1.0, 1.0, 1.0);
    float shininess = 64.0;

    const vec3 ambientLight = 0.5 * vec3(1.0, 1.0, 1.0);
    vec3 color = ambientLight * k_a;
    
    vec3 light1Intensity = vec3(0.4, 0.4, 0.4);
    
    color += phongContribForLight(k_d, k_s, shininess, p,
                                  light1Pos,
                                  light1Intensity);
    
    vec3 light2Intensity = vec3(0.4, 0.4, 0.4);
    
    color += phongContribForLight(k_d, k_s, shininess, p,
                                  light2Pos,
                                  light2Intensity);    
    return color;
}

vec3 lambertContribForLight(vec3 k_d, vec3 p,
                          vec3 lightPos, vec3 lightIntensity) {
    vec3 N = estimateToastsNormal(p);
    vec3 L = normalize(lightPos - p);
    
    float dotLN = dot(L, N);
    
    if (dotLN < 0.0) {
        return vec3(0.0, 0.0, 0.0);
    } 

    return lightIntensity * (k_d * dotLN);
}

vec3 toastLambert(vec3 p) {
    float PI = 3.14159;
	float x = fract(u_Time) * (2.0 * PI);
    /*
    float toastHeight = 1.0 + -((2.0*floor(x/(2.0*PI)) - floor(2.0*(x/(2.0*PI))) + 1.0)*(sin(-(x-(PI/2.0)))+1.0) + 
		2.0*fract(x/(PI / 2.0)) * (2.0*floor((x-PI)/(2.0*PI)) - floor(2.0*((x-PI)/(2.0*PI))) + 1.0) * 
		(2.0*floor((x+(PI/2.0))/(PI)) - floor(2.0*((x+(PI/2.0))/(PI))) + 1.0));*/
    float toastColor = fract(x / (PI)) * (2.0*floor((x + PI) /(2.0*PI)) - floor(2.0*((x + PI)/(2.0*PI))) + 1.0);
    vec3 lightBrown = vec3(1.0, 0.8, 0.6);
    vec3 darkBrown = vec3(0.3, 0.24, 0.18);
    vec3 k_a = vec3(0.1, 0.1, 0.1);
    vec3 k_d = mix(lightBrown, darkBrown, toastColor);
    
    const vec3 ambientLight = 0.5 * vec3(1.0, 1.0, 1.0);
    vec3 color = ambientLight * k_a;
    
    vec3 light1Intensity = vec3(0.4, 0.4, 0.4);
    
    color += lambertContribForLight(k_d, p, light1Pos, light1Intensity);
    
    vec3 light2Intensity = vec3(0.4, 0.4, 0.4);
    
    color += lambertContribForLight(k_d, p,
                                  light2Pos,
                                  light2Intensity);    
    return color;
}

Intersection sceneIntersection(vec3 p) {
	Intersection i;
	i.t = sceneSDF(p);
	i.normal = estimateSceneNormal(p);
	i.color = toasterPhong(p);
	i.id = 1;

	return i;
}

Intersection toastIntersection(vec3 p) {
	Intersection i;
	i.t = animatedToastsSDF(p);
	i.normal = estimateToastsNormal(p);
	i.color = toastLambert(p);
	i.id = 2;

	return i;
}

float toastAndSceneSDF(vec3 p) {
	return unionSDF(sceneSDF(p), animatedToastsSDF(p));
}

// returns union of toast and scene
Intersection toastAndSceneIntersection(vec3 p) {
	Intersection scene = sceneIntersection(p);
	Intersection toast = toastIntersection(p);

	Intersection final;

	Intersection minI = scene;
	if (scene.t > toast.t) {
		minI = toast;
	}

	final.t = minI.t;
	final.normal = minI.normal;
	final.color = minI.color;
	final.id = minI.id;

	return final;

}

float shortestDistanceToSurface(vec3 origin, vec3 marchingDirection, float start, float end) {
    float depth = start;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        float dist = toastAndSceneSDF(origin + depth * marchingDirection);
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
            
vec3 getRayDirection(float fieldOfView, vec2 size, vec2 fragCoord) {
    vec2 xy = fragCoord - size / 2.0;
    float z = size.y / tan(radians(fieldOfView) / 2.0);
    return normalize(vec3(xy, -z));
}

mat4 getViewMatrix(vec3 eye, vec3 center, vec3 up) {
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

mat2 rotate2D(float angle) { 
    return mat2(cos(angle), -sin(angle), sin(angle), cos(angle)); 
}

void main() {
    // frag coord now represents window width/height
	vec2 fragCoord = ((fs_Pos.xy + 1.0) / 2.0) * u_AspectRatio.xy;
	
	vec3 viewDir = getRayDirection(45.0, u_AspectRatio.xy, fragCoord);
    
    mat4 viewToWorld = getViewMatrix(eye, vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));
    vec3 worldDir = (viewToWorld * vec4(viewDir, 0.0)).xyz;
    
    float dist = shortestDistanceToSurface(eye, worldDir, MIN_DIST, MAX_DIST);

    if (dist > MAX_DIST - EPSILON) {
        // Didn't hit anything so show background

        vec2 pos = (fragCoord.xy - 0.5 * u_AspectRatio.xy) / u_AspectRatio.y;
        vec2 rotatedPos = rotate2D(0.6) * pos;
                    
        float xMax = 0.5 * u_AspectRatio.x / u_AspectRatio.y;
        
        // simulate moving stripes in bg
        float stripesMask = step(0.25, mod(rotatedPos.x - u_Time, 0.5));
        
        vec3 stripeColor = vec3(0.3, 0.6, 0.7);
        vec3 frag = stripeColor;

        frag -= 0.1 * stripesMask;	
        frag -= smoothstep(0.45, 2.5, length(pos));

        out_Col = vec4(vec3(frag), 1.0);
		return;
	}
    
    // The closest point on the surface to the eyepoint along the view ray
    vec3 p = eye + dist * worldDir;


    // get intersection of entire scene
	Intersection i = toastAndSceneIntersection(p);
	
    // get color
	vec3 color = i.color;

	// NOW LET'S REFLECT THAT RAY (but not for the toast)
	if (i.id == 1) { // use ID to see if toast or not 
		worldDir = reflect(worldDir, i.normal); // reflect ray
        // send off from intersection point
		dist = shortestDistanceToSurface(p + worldDir * 0.001, worldDir, MIN_DIST, MAX_DIST);
        // calculate new point of intersection
		p = (p + worldDir * 0.001) + dist * worldDir;
		i = toastAndSceneIntersection(p);

        // add reflection color
		color += i.color * 0.35; 
	}
    
    out_Col = vec4(color, 1.0);

}
