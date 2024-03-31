use indicatif::ProgressBar;
use num_traits::float::FloatConst;
use rand::random;
use rayon::prelude::*;
use std::io::Write;
use std::sync::{Arc, Mutex};

type Vec3 = nalgebra::Vector3<f64>;
type HittableList = Vec<Box<dyn Hittable>>;
type MatPointer = Arc<dyn Material>;
static ZERO: Vec3 = Vec3::new(0.0, 0.0, 0.0);
static WORLD_UP: Vec3 = Vec3::new(0.0, 1.0, 0.0);
static INFINITY: f64 = f64::MAX;
static CAMERA_SAMPLE: usize = 512;
static RAY_DEPTH: usize = 5;
static IMAGE_WIDTH: usize = 800;
static IMAGE_HEIGHT: usize = 400;
#[derive(Default)]
struct Ray {
    pub orig: Vec3,
    pub dir: Vec3,
}

impl Ray {
    pub fn new(orig: Vec3, dir: Vec3) -> Self {
        Self { orig, dir }
    }
    pub fn at(&self, t: f64) -> Vec3 {
        self.orig + t * self.dir
    }
}
#[inline]
fn write_color(v: &Vec3) -> String {
    let v = v / (CAMERA_SAMPLE as f64);
    let r = clamp_one(v.x.sqrt());
    let g = clamp_one(v.y.sqrt());
    let b = clamp_one(v.z.sqrt());
    format!(
        "{} {} {}\n",
        (255.999 * r) as u8,
        (255.999 * g) as u8,
        (255.999 * b) as u8
    )
}
#[inline]
fn random_half() -> f64 {
    random::<f64>() - 0.5
}

fn random_in_sphere() -> Vec3 {
    loop {
        let v = Vec3::new(
            (random::<f64>() - 0.5) * 2.0,
            (random::<f64>() - 0.5) * 2.0,
            (random::<f64>() - 0.5) * 2.0,
        );
        if length_squared(&v) < 1.0 {
            return v.normalize();
        }
    }
}
fn random_unit_vector() -> Vec3 {
    let a = random::<f64>() * 2.0 * f64::PI();
    let z = random_half() * 2.0;
    let r = (1.0 - z * z).sqrt();
    Vec3::new(r * a.cos(), r * a.sin(), z)
}
/*
fn random_in_hemisphere(normal: &Vec3) -> Vec3 {
    let in_unit_sphere = random_in_sphere();
    if in_unit_sphere.dot(normal) > 0.0 {
        in_unit_sphere
    } else {
        -in_unit_sphere
    }
}

 */
#[inline]
fn clamp_one(x: f64) -> f64 {
    if x > 0.999 {
        0.999
    } else if x < 0.0 {
        0.0
    } else {
        x
    }
}
#[inline]
fn length_squared(v: &Vec3) -> f64 {
    v.dot(&v)
}
#[inline]
fn length(v: &Vec3) -> f64 {
    v.dot(&v).sqrt()
}
#[inline]
fn reflect(v: &Vec3, n: &Vec3) -> Vec3 {
    v - v.dot(n) * 2.0 * n
}
#[inline]
fn mul(a: &Vec3, b: &Vec3) -> Vec3 {
    Vec3::new(a.x * b.x, a.y * b.y, a.z * b.z)
}

#[inline]
fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * f64::PI() / 180.0
}
fn random_in_unit_disk() -> Vec3 {
    loop {
        let p = Vec3::new(random_half() * 2.0, random_half() * 2.0, 0.0);
        if length_squared(&p) < 1.0 {
            return p;
        }
    }
}
fn random_vec3() -> Vec3 {
    Vec3::new(random::<f64>(), random::<f64>(), random::<f64>())
}
trait Material: Send + Sync {
    fn scatter(
        &self,
        _r_in: &Ray,
        _rec: &HitRecord,
        _attenuation: &mut Vec3,
        _scattered: &mut Ray,
    ) -> bool {
        false
    }
}
#[derive(Default, Copy, Clone)]
struct Lambert {
    pub albedo: Vec3,
}
impl Material for Lambert {
    fn scatter(
        &self,
        _r_in: &Ray,
        rec: &HitRecord,
        attenuation: &mut Vec3,
        scattered: &mut Ray,
    ) -> bool {
        let scatter_direction = rec.normal + random_unit_vector();
        *scattered = Ray::new(rec.p, scatter_direction);
        *attenuation = self.albedo;
        true
    }
}
#[derive(Default, Copy, Clone)]
struct Metal {
    pub albedo: Vec3,
    pub fuzz: f64,
}
impl Material for Metal {
    fn scatter(
        &self,
        r_in: &Ray,
        rec: &HitRecord,
        attenuation: &mut Vec3,
        scattered: &mut Ray,
    ) -> bool {
        let reflected = reflect(&r_in.dir.normalize(), &rec.normal);
        *scattered = Ray::new(rec.p, reflected + self.fuzz * random_in_sphere());
        *attenuation = self.albedo;
        scattered.dir.dot(&rec.normal) > 0.0
    }
}
fn schlick(cosine: f64, ref_idx: f64) -> f64 {
    let mut r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    r0 + (1.0 - r0) * ((1.0 - cosine).powi(5))
}
fn refract(uv: &Vec3, n: &Vec3, etai_over_etat: f64) -> Vec3 {
    let cos_theta = (-uv).dot(n);
    let r_out_parallel = etai_over_etat * (uv + cos_theta * n);
    let r_out_perp = -(1.0 - length_squared(&r_out_parallel)).sqrt() * n;
    r_out_parallel + r_out_perp
}
struct Dielectric {
    ref_idx: f64,
}
impl Material for Dielectric {
    fn scatter(
        &self,
        r_in: &Ray,
        rec: &HitRecord,
        attenuation: &mut Vec3,
        scattered: &mut Ray,
    ) -> bool {
        *attenuation = Vec3::new(1.0, 1.0, 1.0);
        let etai_over_etat = if rec.front_face {
            1.0 / self.ref_idx
        } else {
            self.ref_idx
        };

        let unit_direction = r_in.dir.normalize();
        let cos_theta = (-unit_direction).dot(&rec.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        if etai_over_etat * sin_theta > 1.0 {
            let reflected = reflect(&unit_direction, &rec.normal);
            *scattered = Ray::new(rec.p, reflected);
            return true;
        }
        let reflect_prob = schlick(cos_theta, etai_over_etat);
        if random::<f64>() < reflect_prob {
            let reflected = reflect(&unit_direction, &rec.normal);
            *scattered = Ray::new(rec.p, reflected);
            return true;
        }
        let refracted = refract(&unit_direction, &rec.normal, etai_over_etat);
        *scattered = Ray::new(rec.p, refracted);
        return true;
    }
}
struct HitRecord {
    p: Vec3,
    normal: Vec3,
    mat_ptr: MatPointer,
    t: f64,
    front_face: bool,
}
impl HitRecord {
    pub fn new() -> Self {
        Self {
            p: ZERO,
            normal: ZERO,
            mat_ptr: Arc::new(Lambert {
                ..Default::default()
            }),
            t: 0.0,
            front_face: false,
        }
    }
    pub fn set_face_normal(&mut self, r: &Ray, outward_normal: &Vec3) {
        self.front_face = r.dir.dot(outward_normal) < 0.0;
        self.normal = if self.front_face {
            *outward_normal
        } else {
            -outward_normal
        };
    }
}

trait Hittable: Send + Sync {
    fn hit(&self, _r: &Ray, _t_min: f64, _t_max: f64) -> Option<HitRecord> {
        None
    }
}
struct Sphere {
    pub center: Vec3,
    pub radius: f64,
    pub mat_ptr: MatPointer,
}

impl Hittable for Sphere {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let oc = r.orig - self.center;
        let a = length_squared(&r.dir);
        let half_b = oc.dot(&r.dir);
        let c = length_squared(&oc) - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;
        if discriminant > 0.0 {
            let root = discriminant.sqrt();
            let t = (-half_b - root) / a;
            if t < t_max && t > t_min {
                let mut rec = HitRecord::new();
                rec.t = t;
                rec.p = r.at(t);
                rec.mat_ptr = Arc::clone(&self.mat_ptr);
                let outward_normal = (rec.p - self.center) / self.radius;
                rec.set_face_normal(r, &outward_normal);
                return Some(rec);
            }
            let t = (-half_b + root) / a;
            if t < t_max && t > t_min {
                let mut rec = HitRecord::new();
                rec.t = t;
                rec.p = r.at(t);
                rec.mat_ptr = self.mat_ptr.clone();
                let outward_normal = (rec.p - self.center) / self.radius;
                rec.set_face_normal(r, &outward_normal);
                return Some(rec);
            }
        }
        None
    }
}
fn sky_color(r: &Ray) -> Vec3 {
    let unit_direction = r.dir.normalize();
    let t = 0.5 * (unit_direction.y + 1.0);
    (1.0 - t) * Vec3::new(1.0, 1.0, 1.0) + t * Vec3::new(0.5, 0.7, 1.0)
}
fn hit_world(r: &Ray, world: &HittableList) -> Option<HitRecord> {
    let mut rec_closest = None;
    for w in world {
        if let Some(rec) = w.hit(r, 0.0001, INFINITY) {
            if rec_closest.is_none() {
                rec_closest = Some(rec);
            } else {
                if rec.t < rec_closest.as_ref().unwrap().t {
                    rec_closest = Some(rec);
                }
            }
        }
    }
    rec_closest
}

fn ray_color(r: &Ray, world: &Vec<Box<dyn Hittable>>, depth: usize) -> Vec3 {
    if depth == 0 {
        return ZERO;
    }
    if let Some(rec) = hit_world(&r, &world) {
        let mut scattered = Ray {
            ..Default::default()
        };
        let mut attenuation = Vec3::new(0.0, 0.0, 0.0);
        if rec
            .mat_ptr
            .scatter(&r, &rec, &mut attenuation, &mut scattered)
        {
            return mul(&attenuation, &ray_color(&scattered, world, depth - 1));
        }
        return ZERO;
    } else {
        sky_color(&r)
    }
}

#[derive(Default)]
struct Camera {
    pub origin: Vec3,
    pub lower_left_corner: Vec3,
    pub horizontal: Vec3,
    pub vertical: Vec3,
    pub u: Vec3,
    pub v: Vec3,
    pub w: Vec3,
    lens_radius: f64,
}
impl Camera {
    fn new(
        look_from: Vec3,
        look_at: Vec3,
        vup: Vec3,
        fov: f64,
        aspect: f64,
        aperture: f64,
        focus_dist: f64,
    ) -> Self {
        let mut cam = Camera {
            ..Default::default()
        };
        cam.origin = look_from;
        cam.lens_radius = aperture / 2.0;
        let theta = degrees_to_radians(fov);
        let half_height = (theta / 2.0).tan();
        let half_width = aspect * half_height;
        cam.w = (look_from - look_at).normalize();
        cam.u = vup.cross(&cam.w).normalize();
        cam.v = cam.w.cross(&cam.u);
        cam.lower_left_corner = cam.origin
            - half_width * focus_dist * cam.u
            - half_height * focus_dist * cam.v
            - focus_dist * cam.w;
        cam.horizontal = 2.0 * half_width * focus_dist * cam.u;
        cam.vertical = 2.0 * half_height * focus_dist * cam.v;
        cam
    }
    fn get_ray(&self, s: f64, t: f64) -> Ray {
        let rd = self.lens_radius * random_in_unit_disk();
        let offset = self.u * rd.x + self.v * rd.y;

        Ray::new(
            self.origin + offset,
            self.lower_left_corner + s * self.horizontal + t * self.vertical - self.origin - offset,
        )
    }
}
fn random_scene() -> HittableList {
    let mut world = HittableList::new();

    world.push(Box::new(Sphere {
        center: Vec3::new(0.0, -1000.0, 0.0),
        radius: 1000.0,
        mat_ptr: Arc::new(Lambert {
            albedo: Vec3::new(0.5, 0.5, 0.5),
        }),
    }));
    for a in -11..11 {
        let a = a as f64;
        for b in -11..11 {
            let b = b as f64;
            let choose_mat = random::<f64>();
            let center = Vec3::new(a + 0.9 * random::<f64>(), 0.2, b + 0.9 * random::<f64>());
            if length(&(center - Vec3::new(4.0, 0.2, 0.0))) > 0.9 {
                if choose_mat < 0.8 {
                    let albedo = random_vec3();
                    world.push(Box::new(Sphere {
                        center,
                        radius: 0.2,
                        mat_ptr: Arc::new(Lambert { albedo }),
                    }))
                } else if choose_mat < 0.95 {
                    let albedo = random_vec3() / 2.0 + Vec3::new(0.5, 0.5, 0.5);
                    let fuzz = random::<f64>() / 2.0;
                    world.push(Box::new(Sphere {
                        center,
                        radius: 0.2,
                        mat_ptr: Arc::new(Metal { albedo, fuzz }),
                    }))
                } else {
                    world.push(Box::new(Sphere {
                        center,
                        radius: 0.2,
                        mat_ptr: Arc::new(Dielectric { ref_idx: 1.5 }),
                    }))
                }
            }
        }
    }
    world.push(Box::new(Sphere {
        center: Vec3::new(0.0, 1.0, 0.0),
        radius: 1.0,
        mat_ptr: Arc::new(Dielectric { ref_idx: 1.5 }),
    }));
    world.push(Box::new(Sphere {
        center: Vec3::new(-4.0, 1.0, 0.0),
        radius: 1.0,
        mat_ptr: Arc::new(Lambert {
            albedo: Vec3::new(0.4, 0.2, 0.1),
        }),
    }));
    world.push(Box::new(Sphere {
        center: Vec3::new(4.0, 1.0, 0.0),
        radius: 1.0,
        mat_ptr: Arc::new(Metal {
            albedo: Vec3::new(0.7, 0.6, 0.5),
            fuzz: 0.0,
        }),
    }));
    world
}

async fn progress_manager(progress: Arc<Mutex<usize>>) {
    let mut delay = tokio::time::interval(std::time::Duration::from_secs(1));
    let pb = ProgressBar::new(100);
    loop {
        delay.tick().await;

        let progress = *progress.lock().unwrap();

        if progress >= IMAGE_HEIGHT * IMAGE_WIDTH - 1 {
            pb.finish();
            break;
        } else {
            let pos = (progress as f64 / (IMAGE_HEIGHT * IMAGE_WIDTH) as f64 * 100.0) as u64;
            pb.set_position(pos);
        }
    }
}

#[tokio::main]
async fn main() {
    let mut file = std::fs::File::create("pic.ppm").expect("create failed");
    write!(file, "P3\n{IMAGE_WIDTH} {IMAGE_HEIGHT}\n255\n").expect("write failed");
    let world = random_scene();
    let aspect_ratio: f64 = IMAGE_WIDTH as f64 / IMAGE_HEIGHT as f64;
    let look_from = Vec3::new(13.0, 2.0, 3.0);
    let look_at = Vec3::new(0.0, 0.0, 0.0);
    let dist_to_focus = 10.0;
    let aperture = 0.1;
    let cam = Camera::new(
        look_from,
        look_at,
        WORLD_UP,
        20.0,
        aspect_ratio,
        aperture,
        dist_to_focus,
    );
    let mut output: Vec<Vec3> = vec![ZERO; IMAGE_HEIGHT * IMAGE_WIDTH];
    let progress = Arc::new(Mutex::new(0usize));
    let timer = tokio::spawn(progress_manager(progress.clone()));
    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(x, new_color)| {
            let j = x / IMAGE_WIDTH;
            let i = x % IMAGE_WIDTH;
            for _ in 0..CAMERA_SAMPLE {
                let u = (i as f64 + random_half()) / IMAGE_WIDTH as f64;
                let v = (j as f64 + random_half()) / IMAGE_HEIGHT as f64;
                let r = cam.get_ray(u, v);
                let color = ray_color(&r, &world, RAY_DEPTH);
                *new_color += color;
            }
            *progress.lock().unwrap() += 1;
        });

    for j in (0..IMAGE_HEIGHT).rev() {
        for i in 0..IMAGE_WIDTH {
            write!(file, "{}", write_color(&output[j * IMAGE_WIDTH + i])).unwrap();
        }
    }
    timer.await.unwrap();
    println!("Finish!");
}
