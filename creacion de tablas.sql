--creacion de tabla de cantidad de seguidores por usuario

select id_usuario,count(id_usuario_seguido) as cant_seguidores
into usuarios_seguidores
from siguiendo
group by id_usuario;

--tabla de datos de restaurantes

select id_restaurante,localidad,cocina,precio,
	   case when telefono is not null then 1 else 0 end as telefono,
       case when char_length(rating_comida)>4 then 
       cast(substr(rating_comida,1,2) as numeric)/30 
       else cast(substr(rating_comida,1,1) as numeric)/30 end comida_oleo,
       case when char_length(rating_servicio)>4 then 
       cast(substr(rating_servicio,1,2) as numeric)/30 
       else cast(substr(rating_servicio,1,1) as numeric)/30 end servicio_oleo,
       case when char_length(rating_ambiente)>4 then 
       cast(substr(rating_ambiente,1,2) as numeric)/30 
       else cast(substr(rating_ambiente,1,1) as numeric)/30 end ambiente_oleo,
       fotos
into rest_campos
from restaurantes

-- base para entrenar rf

select a.*,b.edad,b.fecha_alta,b.genero,b.tipo, 
	   c.localidad,c.cocina,c.precio,c.telefono, c.comida_oleo,c.servicio_oleo,c.ambiente_oleo,
       c.fotos
into train_completo_v1
from
ratings_train a,usuarios b,rest_campos c
where a.id_usuario=b.id_usuario
and cast(a.id_restaurante as text)=c.id_restaurante

-- saco los clientes que comentaron 1 sola vez

select * 
into ratings_train_reducido
from ratings_train
where id_usuario in (select id_usuario from (select id_usuario,count(id_usuario)
from ratings_train
group by id_usuario
having count(id_usuario)>4) a)

 -- saco los clientes que comentaron menos de 4 veces y que tienen mas de 3 en algun rating

 select * 
into ratings_train_reducido_nomayor3
from ragtings_train_nomayor3
where id_usuario in (select id_usuario from (select id_usuario,count(id_usuario)
from ragtings_train_nomayor3
group by id_usuario
having count(id_usuario)>4) a)

---sacamos los restaurantes duplicados

select distinct * 
into restaurantes_v3
from restaurantes_v2

---tabla para sacar duplicados con python

select *,"Ir en pareja"+ "Ir con amigos"+ "Comer con buenos tragos"+ "Llevar extranjeros"+ "Escuchar música"+ "Comer sin ser visto"+ "Comer al aire libre"+ "Comer solo"+ "Reunión de negocios"+ "Salida de amigas"+ "Comer bien gastando poco"+ "Ir con la familia"+ "Comer tarde"+ "Comer sano "+ "Merendar"+ "Comer mucho"+ "Ir con chicos"+ "American Express"+ "Cabal"+ "Diners"+ "Electrón"+ "Maestro"+ "Mastercard"+ "Sólo Efectivo"+ "Tarjeta Naranja"+ "Visa" as suma_filtro
into restaurantes_v4
from restaurantes_v3
order by suma_filtro

---tabla restaurantes para joinear v2

select id_restaurante, localidad, cocina, precio, latitud, longitud, fotos, premios, "Ir en pareja", "Ir con amigos", "Comer con buenos tragos", "Llevar extranjeros", "Escuchar música", "Comer sin ser visto", "Comer al aire libre", "Comer solo", "Reunión de negocios", "Salida de amigas", "Comer bien gastando poco", "Ir con la familia", "Comer tarde", "Comer sano ", "Merendar", "Comer mucho", "Ir con chicos", "American Express", "Cabal", "Diners", "Electrón", "Maestro", "Mastercard", "Tarjeta Naranja", "Visa"
	   ,case when telefono is not null then 1 else 0 end as telefono,       
	   case when char_length(rating_comida)>4 then 
       cast(substr(rating_comida,1,2) as numeric)/30 
       else cast(substr(rating_comida,1,1) as numeric)/30 end comida_oleo,
       case when char_length(rating_servicio)>4 then 
       cast(substr(rating_servicio,1,2) as numeric)/30 
       else cast(substr(rating_servicio,1,1) as numeric)/30 end servicio_oleo,
       case when char_length(rating_ambiente)>4 then 
       cast(substr(rating_ambiente,1,2) as numeric)/30 
       else cast(substr(rating_ambiente,1,1) as numeric)/30 end ambiente_oleo
into rest_campos_v2
from restaurantes_v3

---restos dsitintos----

select distinct * into rest_campos_v3 from rest_campos_v2



---tabla final de training rf

select a.*,b.edad,b.fecha_alta,b.genero,b.tipo, 
	   c.localidad, cocina, precio, c.latitud, c.longitud, fotos, premios, "Ir en pareja", "Ir con amigos", "Comer con buenos tragos", "Llevar extranjeros", "Escuchar música", "Comer sin ser visto", "Comer al aire libre", "Comer solo", "Reunión de negocios", "Salida de amigas", "Comer bien gastando poco", "Ir con la familia", "Comer tarde", "Comer sano ", "Merendar", "Comer mucho", "Ir con chicos", "American Express", "Cabal", "Diners", "Electrón", "Maestro", "Mastercard", "Tarjeta Naranja", "Visa",
       telefono,comida_oleo,servicio_oleo,ambiente_oleo
into train_completo_v2
from
ragtings_train a,usuarios b,rest_campos_v2 c
where a.id_usuario=b.id_usuario
and cast(a.id_restaurante as text)=c.id_restaurante


--reduzco para knn

select * 
into ratings_train_reducido
from ratings_train
where id_usuario in (select id_usuario from (select id_usuario,count(id_usuario)
from ratings_train
group by id_usuario
having count(id_usuario)>4) a)
