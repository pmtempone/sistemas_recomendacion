select * from ratings_train;

select * from ratings;

select id_usuario,id_restaurante,substr(fecha,1,4) as anio,
	   substr(fecha,6,2) as mes,
       rating_ambiente,rating_comida,rating_servicio
from ratings_train;

select edad,count(*) from usuarios
group by edad;

select distinct substr(fecha_alta,1,4) as anio from usuarios;

select genero, count(*) from usuarios group by genero;